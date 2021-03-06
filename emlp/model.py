import tensorflow as tf
import numpy as np
from molmod.units import angstrom, electronvolt, pascal
from scipy.optimize import minimize
from .reference import ConstantReference, ConstantFragmentsReference
from .help_functions import load_xyz, filter_cores, MinimizeHistory
import pickle

import os
cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')

 
class Model(tf.Module):
    def __init__(self, cutoff, restore_file = None, float_type = 32, longrange_compute = None, reference = 0.0, xla = False):
        super(Model, self).__init__()
        self.cutoff = cutoff
        self.restore_file = restore_file
        self.longrange_compute = longrange_compute
        self.reference = reference
        if type(self.reference) == str:
            self.reference = ConstantFragmentsReference(self.reference)
        elif type(self.reference) == float:
            self.reference = ConstantReference(self.reference)
        
        if float_type == 32:
            self.float_type = tf.float32
        elif float_type == 64:
            self.float_type = tf.float64
        else:
            raise RuntimeError('Float type %d not implemented.' % float_type)
            
        if xla:
            print('Enabling XLA')
            self.compute_properties = self.xla_compute_properties
        else:
            self.compute_properties = self.default_compute_properties
                  
            
    @classmethod
    def from_restore_file(cls, restore_file, **kwargs):
        ''' kwargs may include:
                float_type (32 or 64)
                reference (default is 0.0)
                longrange_compute (default is None)
                xla (default is False)
        '''
        kwargs.update(pickle.load(open(restore_file + '.pickle', 'rb')))
        kwargs['restore_file'] = restore_file
        my_model = cls(**kwargs)
        my_model.load_checkpoint()
        return my_model
            
                   
    def load_checkpoint(self, ckpt = None):
        if self.restore_file is not None:
            if ckpt is None:
                ckpt = tf.train.Checkpoint(model = self)
            status = ckpt.restore(self.restore_file)
            try:
                status.assert_consumed() # Do some basis checks
                print('The model was successfully restored from file %s' % self.restore_file)
            except Exception as e:
                print(e)
        else:
            print('Starting from random parameters.')

        self.reference.initialize(self)
        
    
    @tf.function(autograph = False, reduce_retracing = True)
    def default_compute_properties(self, inputs, list_of_properties):
        return self._compute_properties(inputs, list_of_properties)
        
        
    @tf.function(autograph = False, reduce_retracing = True, jit_compile = True)
    def xla_compute_properties(self, inputs, list_of_properties):
        return self._compute_properties(inputs, list_of_properties)
        
    
    def _compute_properties(self, inputs, list_of_properties):
        input_rvec = inputs['rvec']
        input_positions = inputs['all_positions']
        input_numbers = inputs['all_numbers']
        input_pairs = inputs['pairs']
        input_efield = inputs['efield']

        if not 'longrange_pairs' in inputs.keys():
            input_lr_pairs = input_pairs
        else:
            input_lr_pairs = inputs['longrange_pairs']

        strain = tf.zeros([tf.shape(input_pairs)[0], 3, 3], dtype = self.float_type)
        
        with tf.GradientTape(persistent = True) as force_tape:
            force_tape.watch(strain)
            force_tape.watch(input_positions)

            input_positions += tf.linalg.matmul(input_positions, strain)
            input_rvec += tf.linalg.matmul(input_rvec, strain)
            
            positions, numbers, rvec, efield, pairs, lr_pairs = self.reference.pad_inputs(input_positions, input_numbers, input_rvec, input_efield, input_pairs, input_lr_pairs)
            self.batches = tf.shape(pairs)[0]
            self.N = tf.shape(pairs)[1]
            
            ''' The shortrange contributions '''
            masks = self.compute_masks(numbers, pairs)
            gather_center, gather_neighbor = self.make_gather_list(pairs, masks['neighbor_mask_int'])
            dcarts, dists = self.compute_distances(positions, numbers, rvec, pairs, masks['neighbor_mask_int'],
                                                   gather_center, gather_neighbor)
                                                                    
            energy, atomic_properties = self.internal_compute(dcarts, dists, numbers, masks, gather_neighbor)
            charges = self.get_charges(numbers, float_type = self.float_type)
            
            if not self.longrange_compute is None:
                lr_inputs = {'charges' : charges, 'positions' : positions, 'elements_mask' : masks['elements_mask'], 'rvec' : rvec}
                if 'longrange_pairs' in inputs.keys():
                    lr_masks = self.compute_masks(numbers, lr_pairs)
                    lr_gather_center, lr_gather_neighbor = self.make_gather_list(lr_pairs, lr_masks['neighbor_mask_int'])
                    lr_dcarts, lr_dists = self.compute_distances(positions, numbers, rvec, lr_pairs, lr_masks['neighbor_mask_int'], lr_gather_center, lr_gather_neighbor)
                    neighbor_charges = self.get_neighbor_charges(charges, lr_gather_neighbor, lr_masks['neighbor_mask'])
                    lr_inputs.update({'neighbor_charges' : neighbor_charges, 'lr_dists' : lr_dists, 'neighbor_mask' : lr_masks['neighbor_mask']})
                if 'n_grid' in inputs.keys():
                    lr_inputs['n_grid'] = inputs['n_grid']
                energy += self.longrange_compute(lr_inputs, float_type = self.float_type) / electronvolt
            
            energy += - tf.reduce_sum(positions * tf.expand_dims(tf.cast(charges, dtype = self.float_type), [-1]) * tf.expand_dims(efield, [1]), [1, 2]) * angstrom / electronvolt
            
            if not 'skip_references' in list_of_properties:
                energy = self.reference.compute_references(energy, numbers, masks, max_N = tf.shape(input_positions)[1])
        
        calculated_properties = {'energy' : energy}
        
        if 'forces' in list_of_properties:
            model_gradient = force_tape.gradient(energy, input_positions)
            calculated_properties['all_forces'] = -model_gradient # These are not masked yet!
        
        if 'vtens' in list_of_properties or 'stress' in list_of_properties:
            vtens = force_tape.gradient(energy, strain) # eV
            calculated_properties['vtens'] = vtens # eV
            volume = tf.reduce_sum(input_rvec[:, :, 0] * tf.linalg.cross(input_rvec[:, :, 1], input_rvec[:, :, 2]), [1])
            calculated_properties['stress'] = vtens / tf.reshape(volume, [-1, 1, 1]) * (electronvolt / angstrom**3) / (1e+09 * pascal) # GPa
            # TODO CORRECT FOR THE EXTERNAL FIELD
            
        if 'masks' in list_of_properties:
            calculated_properties.update(masks)

        return calculated_properties
        
    
    @tf.function(autograph = False, experimental_relax_shapes = True)  
    def _compute_hessian(self, inputs, include_rvec = True, include_efield = True):
        input_rvec = inputs['rvec']
        input_positions_tmp = inputs['all_positions']
        input_numbers = inputs['all_numbers']
        input_pairs = inputs['pairs']
        input_efield = inputs['efield']
        
        # Collecting all the necessary variables
        self.batches = tf.shape(input_pairs)[0]
        self.N = tf.shape(input_pairs)[1]
        if include_rvec:
            gvecs = tf.linalg.inv(input_rvec)
            fractional = tf.einsum('ijk,ikl->ijl', input_positions_tmp, gvecs)
            variables = [tf.reshape(fractional, [self.batches, -1]), tf.reshape(input_rvec, [self.batches, 9])]
        else:
            variables = [tf.reshape(input_positions_tmp, [self.batches, -1])]
        if include_efield:
            variables.append(input_efield)
        
        all_coords = tf.concat(variables, axis=1)
        F = tf.shape(all_coords)[1] # NUMBER OF DIMENSIONS OF THE HESSIAN
        all_coords = tf.stop_gradient(all_coords)
        
        # Extracting the variables
        if include_rvec:
            rvec = tf.reshape(all_coords[:, self.N*3:self.N*3+9], [self.batches, 3, 3]) 
            fractional = tf.reshape(all_coords[:, :self.N*3], [self.batches, -1, 3])
            positions = tf.einsum('ijk,ikl->ijl', fractional, rvec)
        else:
            positions = tf.reshape(all_coords[:, :self.N*3], [self.batches, -1, 3])
            rvec = tf.expand_dims(tf.eye(3, dtype=self.float_type), [0]) * 100.
        if include_efield: 
            efield = all_coords[:, -3:]
        else:
            efield = tf.zeros([1, 3], dtype=self.float_type)
        
        ''' The shortrange contributions '''
        masks = self.compute_masks(input_numbers, input_pairs)
        gather_center, gather_neighbor = self.make_gather_list(input_pairs, masks['neighbor_mask_int'])
        dcarts, dists = self.compute_distances(positions, input_numbers, rvec, input_pairs, masks['neighbor_mask_int'],
                                               gather_center, gather_neighbor)
                                                                
        energy, atomic_properties = self.internal_compute(dcarts, dists, input_numbers, masks, gather_neighbor)
        charges = self.get_charges(input_numbers, float_type = self.float_type)

        if not self.longrange_compute is None:
            lr_inputs = {'charges' : charges, 'positions' : positions, 'elements_mask' : masks['elements_mask'], 'rvec' : rvec}
            if 'longrange_pairs' in inputs.keys():
                lr_pairs = inputs['longrange_pairs']
                lr_masks = self.compute_masks(input_numbers, lr_pairs)
                lr_gather_center, lr_gather_neighbor = self.make_gather_list(lr_pairs, lr_masks['neighbor_mask_int'])
                lr_dcarts, lr_dists = self.compute_distances(positions, input_numbers, rvec, lr_pairs, lr_masks['neighbor_mask_int'], lr_gather_center, lr_gather_neighbor)
                neighbor_charges = self.get_neighbor_charges(charges, lr_gather_neighbor, lr_masks['neighbor_mask'])
                lr_inputs.update({'neighbor_charges' : neighbor_charges, 'lr_dists' : lr_dists, 'neighbor_mask' : lr_masks['neighbor_mask']})
            if 'n_grid' in inputs.keys():
                lr_inputs['n_grid'] = inputs['n_grid']
            energy += self.longrange_compute(lr_inputs, float_type = self.float_type) / electronvolt
        
        energy += - tf.reduce_sum(positions * tf.expand_dims(tf.cast(charges, dtype = self.float_type), [-1]) * tf.expand_dims(efield, [1]), [1, 2]) * angstrom / electronvolt
        # No need to include reference energies here 
        hessian = tf.hessians(energy, all_coords)[0]
        hessian = tf.reshape(hessian, [F, F])
        return hessian
            
    
    def get_charges(self, all_numbers, float_type = tf.float32):
        screened_numbers = tf.where(tf.not_equal(all_numbers, 1), all_numbers - 2, all_numbers)
        screened_numbers = tf.where(tf.math.greater(all_numbers, 10), all_numbers - 10, screened_numbers)
        charges = tf.where(tf.equal(all_numbers, 99), -2 * tf.ones([self.batches, self.N], dtype = tf.int32), screened_numbers)
        return tf.cast(charges, dtype = float_type)


    def get_neighbor_charges(self, charges, gather_neighbor, neighbor_mask):
        neighbor_charges = tf.gather_nd(charges, gather_neighbor) * neighbor_mask # [batches, N, J]
        return neighbor_charges
        
    
    def make_gather_list(self, pairs, neighbor_mask_int):
        J = tf.shape(pairs)[2]
        batch_indices = tf.tile(tf.reshape(tf.range(self.batches), [-1, 1, 1, 1]), [1, self.N, J, 1])
        index_center = tf.tile(tf.reshape(tf.range(self.N), [1, -1, 1, 1]), [self.batches, 1, J, 1])
        
        # A tensor [batches, N, J, 2] containing the indices of the center atoms
        gather_center = tf.concat([batch_indices, index_center], axis = 3)
        # A tensor [batches, N, J, 2] containing the indices of the neighbors neighbors
        gather_neighbor = tf.concat([batch_indices, tf.expand_dims(pairs[:, :, :, 0], [-1])], axis = 3)

        # Gathering on index -1 should be avoided on CPU when checking bounds
        gather_neighbor *= tf.expand_dims(neighbor_mask_int, [-1]) 
        
        return gather_center, gather_neighbor
        
    
    def compute_masks(self, numbers, pairs):
        neighbor_mask_int = tf.cast(tf.not_equal(pairs[:, :, :, 0], -1), tf.int32)          # shape [batches, N, J]
        neighbor_mask = tf.cast(tf.not_equal(pairs[:, :, :, 0], -1), self.float_type)       # shape [batches, N, J]
        elements_mask = tf.cast(numbers > 0, self.float_type)                               # shape [batches, N]
        electron_number_mask = tf.cast(tf.equal(numbers, 99), dtype = self.float_type)
        position_number_mask = tf.cast(tf.not_equal(numbers, 99), dtype = self.float_type) * elements_mask
        return {'neighbor_mask_int' : neighbor_mask_int, 'neighbor_mask' : neighbor_mask, 'elements_mask' : elements_mask,
                'electron_number_mask' : electron_number_mask, 'position_number_mask' : position_number_mask}

                
    def compute_distances(self, positions, numbers, rvecs, pairs, neighbor_mask_int, gather_center, gather_neighbor):     
        # Making an [batches, N, 3, 3] rvecs
        rvecs_matmul = tf.tile(tf.expand_dims(rvecs, [1]), [1, self.N, 1, 1])
        
        # Computing the relative vectors for each pair
        dcarts = tf.add(
            tf.subtract(
                tf.gather_nd(positions, gather_neighbor),
                tf.gather_nd(positions, gather_center)
            ), tf.matmul(tf.cast(pairs[:, :, :, 1:], dtype = self.float_type), rvecs_matmul))

        # Avoid dividing by zero when calculating derivatives
        zero_division_mask = tf.cast(1 - neighbor_mask_int, self.float_type)
        dcarts += tf.expand_dims(zero_division_mask, [-1])
        
        # Computing the squared distances
        dists = tf.sqrt(tf.reduce_sum(tf.square(dcarts), [-1]) + 1e-20)

        return dcarts, dists
        
        
    def save(self, output_file):
        raise NotImplementedError
        
        
    def preprocess(self, positions, numbers, centers, efield, rvec):
        # First we convert the numpy arrays into real tensor with the correct data type
        tf_rvec = tf.convert_to_tensor(rvec, dtype = self.float_type)
        tf_positions = tf.convert_to_tensor(positions, dtype = self.float_type)
        tf_numbers = tf.convert_to_tensor(numbers, dtype = tf.int32)
        tf_centers = tf.convert_to_tensor(centers, dtype = self.float_type)
        tf_efield = tf.convert_to_tensor(efield, dtype = self.float_type)
        
        all_positions = tf.concat([tf_positions, tf_centers], axis = 0)
        all_numbers = tf.concat([numbers, 99 * tf.ones([np.shape(centers)[0]], dtype = tf.int32)], axis = 0)
        
        if self.float_type == tf.float64:
            tf_pairs = cell_list_op.cell_list(tf.cast(all_positions, dtype = tf.float32), tf.cast(tf_rvec, dtype = tf.float32), np.float32(self.cutoff))
        else:
            tf_pairs = cell_list_op.cell_list(all_positions, tf_rvec, np.float32(self.cutoff))
        
        # Pad each input
        inputs = {'all_numbers' : tf.expand_dims(all_numbers, [0]), 'all_positions' : tf.expand_dims(all_positions, [0]),
                  'rvec' : tf.expand_dims(tf_rvec, [0]), 'pairs' : tf.expand_dims(tf_pairs, [0]),
                  'efield' : tf.expand_dims(tf_efield, [0])}
                  
        if not self.longrange_compute is None:
            lr_outputs = self.longrange_compute.preprocess({'all_positions' : all_positions, 'rvec' : tf_rvec}, float_type = self.float_type)
            for key in lr_outputs.keys():
                inputs[key] = tf.expand_dims(lr_outputs[key], [0])
                
        return inputs
        
        
    def compute_static(self, positions, numbers, centers, efield = [0, 0, 0], rvec = 100 * np.eye(3), list_of_properties = ['energy', 'forces']):
        ''' Returns the energy and forces'''
        inputs = self.preprocess(positions, numbers, centers, efield, rvec)
        tf_calculated_properties = self.compute_properties(inputs, list_of_properties)
        
        calculated_properties = {}
        for key in tf_calculated_properties.keys():
            value = tf_calculated_properties[key].numpy()[0]
            
            if key == 'all_forces':
                calculated_properties['forces'] = value[:positions.shape[0], :]
                calculated_properties['center_forces'] = value[positions.shape[0]:, :]
            elif key == 'energy':
                calculated_properties['energy'] = value
            else:
                calculated_properties[key] = value
                
        if 'vtens' in calculated_properties.keys() or 'stress' in calculated_properties.keys(): # The external field should NOT be included in the stress
            all_centers = np.concatenate((positions[np.where(numbers != 1)], centers), axis = 0)
            dipole_vector = np.sum(np.expand_dims(numbers, 1) * positions, axis = 0) - 2 * np.sum(all_centers, axis = 0)
            ext_vtens = - np.expand_dims(dipole_vector * angstrom, -1) * np.expand_dims(efield, 0) / electronvolt
            calculated_properties['vtens'] -= ext_vtens
            calculated_properties['stress'] = calculated_properties['vtens'] / np.linalg.det(rvec) * (electronvolt / angstrom**3) / (1e+09 * pascal) # GPa

        return calculated_properties
        
    
    def compute_hessian(self, positions, numbers, centers, efield = [0, 0, 0], rvec = 100 * np.eye(3), include_rvec = True, include_efield = True):
        ''' Returns the energy and forces'''
        inputs = self.preprocess(positions, numbers, centers, efield, rvec)
        tf_hessian = self._compute_hessian(inputs, include_rvec = include_rvec, include_efield = include_efield)
        return tf_hessian.numpy()
        
        
    def compute(self, positions, numbers, init_centers, rvec = 100 * np.eye(3), efield = [0, 0, 0], max_disp = None, error_on_fail = False, maxiter = 1000, list_of_properties = ['energy', 'forces'], verbose = False):
        history = MinimizeHistory()
        def cost(centers):
            output = self.compute_static(positions, numbers, centers.reshape([-1, 3]), efield, rvec, list_of_properties = ['energy', 'forces', 'skip_references'])
            energy = output['energy']
            gcenter = -output['center_forces']
            history.update(energy, centers.reshape([-1, 3]), np.max(gcenter))
            if self.float_type == tf.float32:
                energy = np.float64(energy)
                gcenter = np.float64(gcenter) 
            return energy, gcenter.flatten()
            
        if not max_disp is None:
            lower = init_centers.flatten() - max_disp
            upper = init_centers.flatten() + max_disp
            bounds = list(zip(lower, upper))
            result = minimize(cost, init_centers.flatten(), jac = True, options = {'maxiter' : maxiter}, bounds = bounds, method = 'L-BFGS-B')
            jac_evaluations = result.nfev
        else:
            result = minimize(cost, init_centers.flatten(), jac = True, options = {'maxiter' : maxiter})
            jac_evaluations = result.njev
        
        if not result.success:
            if error_on_fail:
                raise RuntimeError('Optimization failed')   
            centers, step = history.get_minimum_grad()
            print('Minimization did fail! Taking the minimum gradient step of step %d.' % step)    
        else:
            centers = result.x.reshape([-1, 3])
        if verbose:
            print('Number of jacobian evaluations: %d' % jac_evaluations)  
        
        output = self.compute_static(positions, numbers, centers, efield, rvec, list_of_properties = list_of_properties)
        output['centers'] = centers

        return output

          
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):
        raise NotImplementedError
