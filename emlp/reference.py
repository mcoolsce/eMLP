import tensorflow as tf
import numpy as np
from molmod.units import angstrom, electronvolt
from .help_functions import load_xyz, filter_cores
import os


cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')


class Reference(object):
    def __init__(self):
        pass
        
        
    def pad_inputs(self, *args):
        return args
        
        
    def initialize(self, model):
        pass
          
        
    def compute_references(self, energy, all_numbers, masks, max_N):
        return energy


class ConstantReference(Reference):
    def __init__(self, value = 0., per_atom = False):
        Reference.__init__(self) 
        self.value = value
        self.per_atom = per_atom
        
        
    def initialize(self, model):
        if self.per_atom:
            print('Using a constant per atom reference of %f' % self.value)
        else:
            print('Using a constant reference of %f' % self.value)
        
        
    def compute_references(self, energy, all_numbers, masks, max_N):
        if self.per_atom:
            return energy + tf.reduce_sum(masks['position_number_mask'], [-1]) * self.value
        else:
            return energy + self.value
        
        
class ConstantFragmentsReference(Reference):
    def __init__(self, label = 'pbe0_aug-cc-pvtz', float_type = tf.float32):
        Reference.__init__(self) 
        self.label = label
        self.float_type = float_type

        ref_data = np.genfromtxt(os.path.dirname(__file__) + '/ref/reference_' + label + '.txt')
        self.ab_initio_energies = ref_data[:, 1] / electronvolt
        
        
    def initialize(self, model):
        print('Constant fragments references at %s' % self.label)
        tmp = []
        for fragment in ['H+', 'H2', 'CH4', 'NH3', 'H2O']:
            for data in load_xyz(os.path.dirname(__file__) + '/ref/' + fragment + '.xyz'):
                all_numbers = data['Z']
                all_positions = data['pos']
                centers = all_positions[np.where(all_numbers == 99)]
                positions = all_positions[np.where(all_numbers != 99)]
                numbers = all_numbers[np.where(all_numbers != 99)]
                centers = filter_cores(centers, positions, numbers)
                output = model.compute_static(positions, numbers, centers, efield = [0, 0, 0], rvec = 100 * np.eye(3), list_of_properties = ['energy', 'skip_references'])
                tmp.append(output['energy'])
                
        self.ref_energies = tf.convert_to_tensor(tmp, dtype=self.float_type) - self.ab_initio_energies
            
        
    def compute_references(self, energy, all_numbers, masks, max_N):
        num_H = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 1), dtype = self.float_type), [-1])
        num_C = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 6), dtype = self.float_type), [-1])
        num_N = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 7), dtype = self.float_type), [-1])
        num_O = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 8), dtype = self.float_type), [-1])
        num_valence = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 99), dtype = self.float_type), [-1])
        num_centers = num_C + num_O + num_N + num_valence # The total number of electron pairs
        Q = num_H + 6 * num_C + 7 * num_N + 8 * num_O - 2 * num_centers # The total charge of the system
        ref_energy = num_C * self.ref_energies[2] + num_N * self.ref_energies[3] + num_O * self.ref_energies[4] + (num_H - 4 * num_C - 3 * num_N - 2 * num_O - Q) / 2. * self.ref_energies[1] + Q * self.ref_energies[0]
        return energy - ref_energy



class ConsistentFragmentsReference(Reference):
    def __init__(self, label = 'pbe0_aug-cc-pvtz', float_type = tf.float32):
        Reference.__init__(self)    
        self.float_type = float_type
        self.label = label
        
        # Preprocessing the input tensors
        self.ref_positions = tf.zeros([0, 9, 3], dtype = float_type)
        self.ref_numbers = tf.zeros([0, 9], dtype = tf.int32)
        self.ref_pairs = tf.zeros([0, 9, 8, 4], dtype = tf.int32)
        
        for fragment in ['H+', 'H2', 'CH4', 'NH3', 'H2O']:
            for data in load_xyz(os.path.dirname(__file__) + '/ref/' + fragment + '.xyz'):
                my_pos, my_numbers, my_pairs = self.load_input(data)
                
                self.ref_positions = tf.concat((self.ref_positions, tf.pad(my_pos, [[0, 0], [0, 9 - tf.shape(my_pos)[1]], [0, 0]], constant_values = 0.)), 0)
                self.ref_numbers = tf.concat((self.ref_numbers, tf.pad(my_numbers, [[0, 0], [0, 9 - tf.shape(my_numbers)[1]]], constant_values = -1)), 0)
                self.ref_pairs = tf.concat((self.ref_pairs, tf.pad(my_pairs, [[0, 0], [0, 9 - tf.shape(my_pairs)[1]], [0, 8 - np.shape(my_pairs)[2]], [0, 0]], constant_values = -1)), 0)
                
                
    def initialize(self, model):
        print('Live consistent fragments references at %s' % self.label)
                        
        
    def load_input(self, data):
        positions = data['pos'][np.where(data['Z'] != 99)]
        centers = data['pos'][np.where(data['Z'] == 99)]
        numbers = data['Z'][np.where(data['Z'] != 99)] 
        centers = filter_cores(centers, positions, numbers)
        all_pos = np.concatenate((positions, centers), axis=0)
        all_num = np.concatenate((numbers, 99 * np.ones(centers.shape[0], dtype=np.int)), axis=0)
        
        all_positions = tf.convert_to_tensor(all_pos, dtype = self.float_type)
        all_numbers = tf.convert_to_tensor(all_num, dtype = tf.int32)

        if self.float_type == tf.float64:
            pairs = cell_list_op.cell_list(tf.cast(all_positions, dtype = tf.float32), tf.eye(3, dtype=self.float_type) * 100, np.float32(16.5))
        else:
            pairs = cell_list_op.cell_list(all_positions, tf.eye(3, dtype=self.float_type) * 100, np.float32(16.5))
        
        return tf.expand_dims(all_positions, [0]), tf.expand_dims(all_numbers, [0]), tf.expand_dims(pairs, [0])
        
        
    def pad_inputs(self, positions, numbers, rvec, efield, pairs, lr_pairs):
        max_N = tf.shape(positions)[1]
        max_J = tf.shape(pairs)[2]
        max_lr_J = tf.shape(lr_pairs)[2]
        
        pad_N_frag = tf.where(max_N - 9 < 0, 0, max_N - 9)
        pad_J_frag = tf.where(max_J - 8 < 0, 0, max_J - 8)
        pad_lr_J_frag = tf.where(max_lr_J - 8 < 0, 0, max_lr_J - 8)
        
        pad_N_all = tf.where(9 - max_N < 0, 0, 9 - max_N)
        pad_J_all = tf.where(8 - max_J < 0, 0, 8 - max_J)
        pad_lr_J_all = tf.where(8 - max_lr_J < 0, 0, 8 - max_lr_J)
        
        positions_padded = tf.pad(positions, [[0, 0], [0, pad_N_all], [0, 0]], constant_values = 0.)
        numbers_padded = tf.pad(numbers, [[0, 0], [0, pad_N_all]], constant_values = -1)
        pairs_padded = tf.pad(pairs, [[0, 0], [0, pad_N_all], [0, pad_J_all], [0, 0]], constant_values = -1)
        lr_pairs_padded = tf.pad(lr_pairs, [[0, 0], [0, pad_N_all], [0, pad_lr_J_all], [0, 0]], constant_values = -1)
        
        positions_ref = tf.concat((positions_padded, tf.pad(self.ref_positions, [[0, 0], [0, pad_N_frag], [0, 0]], constant_values = 0.)), 0)
        numbers_ref = tf.concat((numbers_padded, tf.pad(self.ref_numbers, [[0, 0], [0, pad_N_frag]], constant_values = -1)), 0)
        pairs_ref = tf.concat((pairs_padded, tf.pad(self.ref_pairs, [[0, 0], [0, pad_N_frag], [0, pad_J_frag], [0, 0]], constant_values = -1)), 0)
        lr_pairs_ref = tf.concat((lr_pairs_padded, tf.pad(self.ref_pairs, [[0, 0], [0, pad_N_frag], [0, pad_lr_J_frag], [0, 0]], constant_values = -1)), 0)
        
        rvec_ref = tf.concat((rvec, tf.tile(tf.reshape(tf.eye(3, dtype = self.float_type) * 100., [1, 3, 3]), [5, 1, 1])), 0)
        efield_ref = tf.concat((efield, tf.zeros([5, 3], dtype = self.float_type)), 0)
        
        return positions_ref, numbers_ref, rvec_ref, efield_ref, pairs_ref, lr_pairs_ref
        
    
    def compute_references(self, energy, all_numbers, masks, max_N):
        num_H = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 1), dtype = self.float_type), [-1])
        num_C = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 6), dtype = self.float_type), [-1])
        num_N = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 7), dtype = self.float_type), [-1])
        num_O = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 8), dtype = self.float_type), [-1])
        num_valence = tf.reduce_sum(tf.cast(tf.equal(all_numbers, 99), dtype = self.float_type), [-1])
        num_centers = num_C + num_O + num_N + num_valence
        Q = num_H + 6 * num_C + 7 * num_N + 8 * num_O - 2 * num_centers
        ref_energy = num_C * energy[-3] + num_N * energy[-2] + num_O * energy[-1] + (num_H - 4 * num_C - 3 * num_N - 2 * num_O - Q) / 2. * energy[-4] + Q * energy[-5]
        masks['electron_number_mask'] = masks['electron_number_mask'][:-5,:max_N]
        masks['position_number_mask'] = masks['position_number_mask'][:-5,:max_N]
        masks['elements_mask'] = masks['elements_mask'][:-5,:max_N]
        return (energy - ref_energy)[:-5]
        
        
        
        

