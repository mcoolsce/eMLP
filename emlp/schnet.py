import tensorflow as tf
import numpy as np
from .model import Model
from .help_functions import weight_variable, bias_variable
import pickle
from molmod.units import angstrom


def activation(tensor):
    return tf.math.softplus(tensor) - np.log(2.0)
    

def f_cutoff(input_tensor, cutoff, cutoff_transition_width = 0.5, float_type = tf.float32):
    r_transition = tf.where(input_tensor > cutoff, tf.zeros(tf.shape(input_tensor), dtype = float_type),
                            0.5 * (1 + tf.cos(np.pi * (input_tensor - cutoff + cutoff_transition_width) / cutoff_transition_width)))

    return tf.where(input_tensor > cutoff - cutoff_transition_width, r_transition, tf.ones(tf.shape(input_tensor), dtype = float_type))
    

class InteractionBlock(tf.Module):
    def __init__(self, layer_index, num_features, num_filters, filter_block, float_type = tf.float32):
        super(InteractionBlock, self).__init__()
        
        self.float_type = float_type
        self.num_features = num_features
        self.num_filters = num_filters
        
        self.int_weights = weight_variable([self.num_features, self.num_filters], 'int_weights_%d' % layer_index, stddev = 1. / np.sqrt(self.num_features))
        self.int_bias = bias_variable([1, 1, self.num_filters], 'int_bias_%d' % layer_index)
        
        self.int_weights2 = weight_variable([self.num_filters, self.num_filters], 'int_weights2_%d' % layer_index, stddev = 1. / np.sqrt(self.num_filters))
        self.int_bias2 = bias_variable([1, 1, self.num_filters], 'int_bias2_%d' % layer_index)
        
        self.int_weights3 = weight_variable([self.num_filters, self.num_features], 'int_weights3_%d' % layer_index, stddev = 1. / np.sqrt(self.num_filters))
        self.int_bias3 = bias_variable([1, 1, self.num_features], 'int_bias3_%d' % layer_index)
        
        self.filter_block = filter_block
    
    def __call__(self, atom_features, radial_features, elements_mask, neighbor_mask, smooth_cutoff_mask, gather_neighbor):
        if self.float_type == tf.float64:
            int_weights = tf.cast(self.int_weights, dtype = tf.float64)
            int_bias = tf.cast(self.int_bias, dtype = tf.float64)
            
            int_weights2 = tf.cast(self.int_weights2, dtype = tf.float64)
            int_bias2 = tf.cast(self.int_bias2, dtype = tf.float64)
            
            int_weights3 = tf.cast(self.int_weights3, dtype = tf.float64)
            int_bias3 = tf.cast(self.int_bias3, dtype = tf.float64)
            
        else:
            int_weights = self.int_weights
            int_bias = self.int_bias
            
            int_weights2 = self.int_weights2
            int_bias2 = self.int_bias2
            
            int_weights3 = self.int_weights3
            int_bias3 = self.int_bias3
            
        atom_features = (tf.tensordot(atom_features, int_weights, [[2], [0]]) + int_bias) * tf.expand_dims(elements_mask, [-1])
        neighbor_states = tf.gather_nd(atom_features, gather_neighbor) # [batches, N, J, H]
        
        W = self.filter_block(radial_features) * smooth_cutoff_mask
        
        # Normalizing this sum is important to achieve better training
        # A constant value of 70 is chosen
        conv = tf.reduce_sum(neighbor_states * W, axis = [2]) / 70.
        
        atom_features = activation(tf.tensordot(conv, int_weights2, [[2], [0]]) + int_bias2) * tf.expand_dims(elements_mask, [-1])
        atom_features = (tf.tensordot(atom_features, int_weights3, [[2], [0]]) + int_bias3) * tf.expand_dims(elements_mask, [-1])

        return atom_features
        
        
class FilterBlock(tf.Module):
    def __init__(self, layer_index, n_max, num_filters, float_type = tf.float32):
        super(FilterBlock, self).__init__()
        
        self.n_max = n_max
        self.num_filters = num_filters
        self.float_type = float_type
        
        self.filter_weights1 = weight_variable([self.n_max, self.num_filters], 'filter_weights_1_layer_%d' % layer_index, stddev = 1. / np.sqrt(self.n_max))
        self.filter_bias1 = bias_variable([1, 1, 1, self.num_filters], 'filter_bias_1_layer_%d' % layer_index)
        
        self.filter_weights2 = weight_variable([self.num_filters, self.num_filters], 'filter_weights_2_layer_%d' % layer_index, stddev = 1. / np.sqrt(self.num_filters))
        self.filter_bias2 = bias_variable([1, 1, 1, self.num_filters], 'filter_bias_2_layer_%d' % layer_index)
              
    def __call__(self, radial_features):
        if self.float_type == tf.float64:
            filter_weights1 = tf.cast(self.filter_weights1, dtype = tf.float64)
            filter_bias1 = tf.cast(self.filter_bias1, dtype = tf.float64)
            
            filter_weights2 = tf.cast(self.filter_weights2, dtype = tf.float64)
            filter_bias2 = tf.cast(self.filter_bias2, dtype = tf.float64)
            
        else:
            filter_weights1 = self.filter_weights1
            filter_bias1 = self.filter_bias1
            
            filter_weights2 = self.filter_weights2
            filter_bias2 = self.filter_bias2
            
        dense1 = activation(tf.tensordot(radial_features, filter_weights1, [[3], [0]]) + filter_bias1)   
        return (activation(tf.tensordot(dense1, filter_weights2, [[3], [0]]) + filter_bias2))
        
        
class OutputLayer(tf.Module):
    def __init__(self, prefix_name, layer_sizes, initial_size, float_type = tf.float32):
        super(OutputLayer, self).__init__()
        self.float_type = float_type
        
        previous_size = initial_size
        self.layer_sizes = layer_sizes
        
        self.output_weights = []
        self.output_biases = []
        
        for i, layer_size in enumerate(self.layer_sizes):    
            self.output_weights.append(weight_variable([previous_size, layer_size], prefix_name + '_weights_layer_%d' % i, stddev = 1. / np.sqrt(previous_size)))
            self.output_biases.append(bias_variable([1, 1, layer_size], prefix_name + '_bias_layer_%d' % i))
            
            previous_size = layer_size
                  
    def __call__(self, atom_features):
        for i, layer_size in enumerate(self.layer_sizes):
            w = self.output_weights[i]
            b = self.output_biases[i] 
            
            if self.float_type == tf.float64:
                w = tf.cast(w, dtype = tf.float64)
                b = tf.cast(b, dtype = tf.float64)           
            
            atom_features = tf.tensordot(atom_features, w, [[2], [0]]) + b
            if i < len(self.layer_sizes) - 1: # No activation function for the final layer
                atom_features = activation(atom_features)
                        
        return atom_features


class SchNet(Model):
    def __init__(self, cutoff = 5., n_max = 25, num_features = 64, start = 0.0, num_layers = 3, end = None, num_filters = -1, longrange_compute = None, cutoff_transition_width = None,
                 restore_file = None, float_type = 32, shared_W_interactions = True, reference = 0.0):
        Model.__init__(self, cutoff, restore_file = restore_file, float_type = float_type, longrange_compute = longrange_compute, reference = reference) 
        if end is None:
            self.end = cutoff
        else:
            self.end = end
        self.n_max = n_max
        self.start = start
        self.radial_sigma = (self.end - self.start) / self.n_max
        if cutoff_transition_width is None:
            self.cutoff_transition_width = cutoff - end
        else:
            self.cutoff_transition_width = cutoff_transition_width
        if num_filters == -1:
            self.num_filters = H
        else:
            self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_features = num_features # Size of the hidden state vector
        self.shared_W_interactions = shared_W_interactions

        # Here, all variables are listed and other submodules
        self.init_features = weight_variable([100, self.num_features], 'init_features', stddev = 1.)
        
        if self.shared_W_interactions:
            fixed_W = FilterBlock(0, self.n_max, self.num_filters, self.float_type)
        
        self.interaction_blocks = []
        for layer_index in range(num_layers):
            if self.shared_W_interactions:
                filter_block = fixed_W
            else:
                filter_block = FilterBlock(layer_index, self.n_max, self.num_filters, self.float_type)
                
            self.interaction_blocks.append(InteractionBlock(layer_index, self.num_features, self.num_filters, filter_block, float_type = self.float_type))
            
        self.output_layer = OutputLayer('output_layer', [self.num_features, int(self.num_features / 2), 1], self.num_features, float_type = self.float_type)

       
    def save(self, output_file):
        data = {'cutoff' : self.cutoff, 'n_max' : self.n_max, 'start' : self.start, 'end' : self.end, 'shared_W_interactions' : self.shared_W_interactions, 'cutoff_transition_width' : self.cutoff_transition_width,
                'num_features' : self.num_features, 'num_layers' : self.num_layers, 'num_filters' : self.num_filters}
        pickle.dump(data, open(output_file + '.pickle', 'wb'))
        
        
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):  
        ''' Defining the radial features '''
        n = self.start + tf.range(self.n_max, dtype = self.float_type) / self.n_max * (self.end - self.start)
        n = tf.reshape(n, [1, 1, 1, -1])
        radial_features = tf.exp(- 0.5 * (tf.expand_dims(dists, [-1]) - n)**2 / self.radial_sigma**2) # shape [batches, N, J, n]
        
        # Imposing a smooth cutoff
        smooth_cutoff_mask = f_cutoff(dists, cutoff = self.cutoff, cutoff_transition_width = self.cutoff_transition_width, float_type = self.float_type) # shape [batches, N, J]
        smooth_cutoff_mask = tf.expand_dims(smooth_cutoff_mask * masks['neighbor_mask'], [-1]) # shape [batches, N, J, 1]
            
        if self.float_type == tf.float64:
            init_features = tf.cast(self.init_features, dtype = tf.float64)
        else:
            init_features = self.init_features

        atom_features = tf.nn.embedding_lookup(init_features, numbers * tf.cast(numbers > 0, tf.int32)) * tf.expand_dims(masks['elements_mask'], [-1])
        
        ''' The interaction layers '''
        for i in range(self.num_layers):
            atom_features += self.interaction_blocks[i](atom_features, radial_features, masks['elements_mask'], masks['neighbor_mask'], smooth_cutoff_mask, gather_neighbor)

        atomic_energies = tf.reshape(self.output_layer(atom_features), [self.batches, self.N])
        
        ''' The final energy '''
        energy = tf.reduce_sum(atomic_energies * masks['elements_mask'], [-1])

        return energy, atomic_energies
