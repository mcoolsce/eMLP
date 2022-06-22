import tensorflow as tf
import numpy as np
from molmod.units import angstrom, electronvolt
from .help_functions import load_xyz, XYZLogger, filter_cores
import os


cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))  
    

def f_cutoff(input_tensor, cutoff, cutoff_transition_width = 0.5, float_type = tf.float32):
    r_transition = tf.where(input_tensor > cutoff, tf.zeros(tf.shape(input_tensor), dtype = float_type),
                            0.5 * (1 + tf.cos(np.pi * (input_tensor - cutoff + cutoff_transition_width) / cutoff_transition_width)))

    return tf.where(input_tensor > cutoff - cutoff_transition_width, r_transition, tf.ones(tf.shape(input_tensor), dtype = float_type))


class DataSet(object):
    def __init__(self, tfr_files, num_configs, cutoff = 5., longrange_compute = None, batch_size = 16, test = False, num_shuffle = -1, float_type = 32, 
                 num_parallel_calls = 8, strategy = None, list_of_properties = ['positions', 'numbers', 'energy', 'rvec', 'forces'],
                 additional_property_definitions = {}, augment_data = None):
        self.cutoff = cutoff
        self.longrange_compute = longrange_compute
        self.num_configs = num_configs
        self.list_of_properties = list_of_properties # TODO check if necessary properties are present?
        self.batch_size = batch_size
        self.augment_data = augment_data
        
        if not 'positions' in list_of_properties:
            raise RuntimeError('Positions are required.')
        if not 'numbers' in list_of_properties:
            raise RuntimeError('Numbers are required.')
        
        if float_type == 32:
            self.float_type = tf.float32
            self.zero = 0.        
        elif float_type == 64:
            self.float_type = tf.float64
            self.zero = np.float64(0.)
            
        # For every property, the tuple (shape, pad_index, dtype) is defined.
        self.property_definitions = {'energy': ([1], self.zero, self.float_type),
                                     'positions': ([None, 3], self.zero, self.float_type),
                                     'numbers': ([None], -1, tf.int32),
                                     'pairs': ([None, None, 4], -1, tf.int32),
                                     'longrange_pairs': ([None, None, 4], -1, tf.int32),
                                     'n_grid': ([None, 3], 0, tf.int32),
                                     'rvec': ([3, 3], self.zero, self.float_type),
                                     'vtens' : ([3, 3], self.zero, self.float_type),
                                     'stress' : ([3, 3], self.zero, self.float_type),
                                     'forces' : ([None, 3], self.zero, self.float_type),
                                     'centers' : ([None, 3], self.zero, self.float_type),
                                     'center_forces' : ([None, 3], self.zero, self.float_type),
                                     'efield' : ([3], self.zero, self.float_type)}
        self.property_definitions.update(additional_property_definitions)
        
        padding_shapes = {'pairs': [None, None, 4], 'efield' : [3],
                          'all_positions': [None, 3], 'all_numbers' : [None], 'rvec' : [3, 3]}
        padding_values = {'pairs': -1, 'all_positions': self.zero, 
                          'all_numbers' : -1, 'rvec' : self.zero, 'efield': self.zero}
        if 'forces' in self.list_of_properties:
            padding_shapes['all_forces'] = [None, 3]
            padding_values['all_forces'] = self.zero
        for key in self.list_of_properties:
            if not key in self.property_definitions.keys():
                raise RuntimeError('The property definition of the key "%s" is not known. Use the additional_property_definitions keyword to update its definition.' % key)
            if key in ['positions', 'numbers', 'centers', 'forces', 'center_forces']:
                continue
            padding_shapes[key] = self.property_definitions[key][0]
            padding_values[key] = self.property_definitions[key][1]       
        if not self.longrange_compute is None:
            padding_shapes.update(self.longrange_compute.additional_shapes)
            padding_values.update(self.longrange_compute.additional_values)

        if num_shuffle == -1:
            num_shuffle = self.num_configs
        
        if test:
            print('Initializing test set')
        else:
            print('Initializing training set')
            
        print('Total number of systems found: %d' % self.num_configs)
        print('Initializing the dataset with cutoff radius %f' % self.cutoff)
        print('Using float%d' % float_type)
        print('Batch size: %d' % batch_size)
        print('List of properties: ' + str(self.list_of_properties))
        if not 'rvec' in self.list_of_properties:
            print('WARNING: internally rvec will be set to 100 * np.eye(3) to calculate pairs')
        if not self.augment_data is None:
            self.augment_data.print_info()
        print('')
        
        self._dataset = tf.data.TFRecordDataset(tfr_files)
        
        if test:
            self._dataset = self._dataset.repeat(1) # Repeat only once
        else:
            self._dataset = self._dataset.shuffle(num_shuffle)
            self._dataset = self._dataset.repeat() # Repeat indefinitely
         
        self._dataset = self._dataset.map(self.parser, num_parallel_calls = num_parallel_calls)
        self._dataset = self._dataset.padded_batch(self.batch_size, padded_shapes = padding_shapes, padding_values = padding_values)      
        self._dataset = self._dataset.prefetch(self.batch_size)
        
        if strategy:
            self._dataset = strategy.experimental_distribute_dataset(self._dataset)
            
    
    def parser(self, element):
        # Create the feature vector based on our property list
        feature = {}
        for key in self.list_of_properties:
            feature[key] = tf.io.FixedLenFeature((), tf.string)   
        parsed_features = tf.io.parse_single_example(element, features = feature)
        
        output_dict = {}
        for key in self.list_of_properties:
            output_dict[key] = tf.io.decode_raw(parsed_features[key], self.property_definitions[key][2])
            
        # Replace every None with the number of atoms
        num_atoms = tf.shape(output_dict['numbers'])[0]
        num_centers = tf.shape(output_dict['centers'])[0] // 3 
            
        if not 'rvec' in self.list_of_properties:
            output_dict['rvec'] = 100. * tf.eye(3, dtype=self.float_type)    
            
        if not 'center_forces' in self.list_of_properties :
            output_dict['center_forces'] = tf.zeros([num_centers, 3], dtype=self.float_type)
            
        if not 'efield' in self.list_of_properties:
            output_dict['efield'] = tf.zeros([3], dtype = self.float_type)  
        
        for key in self.list_of_properties:
            if 'center' in key:
                N = num_centers
            else:
                N = num_atoms
            new_shape = [N if dim is None else dim for dim in self.property_definitions[key][0]]
            output_dict[key] = tf.reshape(output_dict[key], new_shape)
            
        if not self.augment_data is None:
            new_centers, aug_forces, aug_center_forces, aug_energy = self.augment_data.augment(output_dict)
            output_dict['centers'] = new_centers
            output_dict['forces'] += aug_forces
            output_dict['center_forces'] += aug_center_forces
            output_dict['energy'] += aug_energy
        
        # We overwrite those entries
        output_dict['all_positions'] = tf.concat([output_dict['positions'], output_dict['centers']], axis = 0)
        output_dict['all_numbers'] = tf.concat([output_dict['numbers'], 99 * tf.ones([num_centers], dtype = tf.int32)], axis = 0)
        if 'forces' in self.list_of_properties:
            output_dict['all_forces'] = tf.concat([output_dict['forces'], output_dict['center_forces']], axis = 0)
            del output_dict['forces']
        
        # Remove unnecessary tensors
        del output_dict['numbers']
        del output_dict['positions']
        del output_dict['centers']
        del output_dict['center_forces']
        
        if self.float_type == tf.float64:
            pairs = cell_list_op.cell_list(tf.cast(output_dict['all_positions'], dtype = tf.float32), tf.cast(output_dict['rvec'], dtype = tf.float32), np.float32(self.cutoff))
        else:
            pairs = cell_list_op.cell_list(output_dict['all_positions'], output_dict['rvec'], np.float32(self.cutoff))
        output_dict['pairs'] = pairs
        
        if not self.longrange_compute is None:
            output_dict.update(self.longrange_compute.preprocess(output_dict, float_type = self.float_type))
        
        return output_dict
        

class DataAugmentation(object):
    def __init__(self, delta_lower = 0.06, delta_upper = 0.12, k = 2, percentage = 0.1, cutoff = 4.0, periodic=False, float_type=tf.float32):
        self.delta_lower = delta_lower # angstrom
        self.delta_upper = delta_upper # angstrom
        self.k = k # Ha / angstrom**2
        self.percentage = percentage
        self.cutoff = cutoff
        self.periodic = periodic
        self.float_type = float_type


    def print_info(self):
        print('Using %.1f%% data augmentation with the following properties:' % (self.percentage * 100))
        print('Delta R in [%f, %f], k=%f, periodic=%s, cutoff=%f' % (self.delta_lower, self.delta_upper, self.k, self.periodic, self.cutoff))


    def _augment(self, data):
        delta_R = tf.random.uniform(shape = [], minval = self.delta_lower, maxval = self.delta_upper, dtype = self.float_type)
        delta_E = 0.5 * self.k * delta_R**2 / electronvolt
        delta_F = self.k * delta_R / electronvolt
        
        centers = data['centers']
        N = tf.shape(centers)[0]
        positions = data['positions']
        efield = data['efield']

        random_center = tf.random.uniform(shape = [], minval = 0, maxval = N, dtype = tf.int32)
        mask = tf.where(tf.equal(tf.range(N, dtype = tf.int32), random_center), tf.ones([N], dtype = self.float_type), tf.zeros([N], dtype = self.float_type))
        
        # Generating a random direction
        u = tf.random.uniform(shape = [], dtype = self.float_type)
        v = tf.random.uniform(shape = [], dtype = self.float_type)
        xdir = tf.sqrt(1 - (1 - 2 * u)**2) * tf.math.sin(2 * np.pi * v)
        ydir = tf.sqrt(1 - (1 - 2 * u)**2) * tf.math.cos(2 * np.pi * v)
        zdir = 1 - 2 * u
        
        direction = tf.stack((xdir, ydir, zdir), axis = 0)
        displacement = direction * delta_R
        
        # Applying the displacement
        centers += tf.expand_dims(mask, [1]) * tf.expand_dims(displacement, [0])
        
        # Calculating the force on the displaced center
        returning_force = - direction * delta_F
        
        my_pos = centers[random_center, :] # New augmented position
        
        if self.periodic: # TODO this calculation is not valid in general. Only well behaving rvecs suffice.
            rvec = data['rvec']
            gvecs = tf.linalg.inv(rvec)
            frac_rel_vector = tf.einsum('jk,kl->jl', tf.expand_dims(my_pos, [0]) - positions, gvecs)
            frac_rel_vector = (frac_rel_vector % 1.) - ((frac_rel_vector % 1.) // 0.5)
            rel_pos = tf.sqrt(tf.reduce_sum(tf.einsum('jk,kl->jl', frac_rel_vector, rvec)**2, [1]))
            frac_rel_vector = tf.einsum('jk,kl->jl', tf.expand_dims(my_pos, [0]) - centers, gvecs)
            frac_rel_vector = (frac_rel_vector % 1.) - ((frac_rel_vector % 1.) // 0.5)
            rel_cen = tf.sqrt(tf.reduce_sum(tf.einsum('jk,kl->jl', frac_rel_vector, rvec)**2, [1]))
        else:
            rel_pos = tf.sqrt(tf.reduce_sum((tf.expand_dims(my_pos, [0]) - positions)**2, [1]))
            rel_cen = tf.sqrt(tf.reduce_sum((tf.expand_dims(my_pos, [0]) - centers)**2, [1]))
        
        division_rel_pos = tf.where(rel_pos < 1e-02, 1e-02 * tf.ones(tf.shape(rel_pos), dtype = self.float_type), rel_pos)
        division_rel_cen = tf.where(rel_cen < 1e-02, 1e-02 * tf.ones(tf.shape(rel_cen), dtype = self.float_type), rel_cen)
        
        position_weights = f_cutoff(rel_pos, self.cutoff, 1.0, self.float_type) / division_rel_pos**2
        center_weights = (1 - mask) * f_cutoff(rel_cen, self.cutoff, 1.0, self.float_type) / division_rel_cen**2
        
        Q_0 = tf.reduce_sum(position_weights) + tf.reduce_sum(center_weights)
        
        if self.periodic: # The rotational moment should not be zero in periodic calculations
            lambd = tf.expand_dims(-returning_force / Q_0, [0])
            correction_pos_force = lambd * tf.expand_dims(position_weights, [1])
            correction_cen_force = lambd * tf.expand_dims(center_weights, [1])
        else:
            Q_1 = tf.reduce_sum(positions * tf.expand_dims(position_weights, [1]), [0]) + tf.reduce_sum(centers * tf.expand_dims(center_weights, [1]), [0])

            translation_vector = - Q_1 / Q_0
            translated_positions = positions + tf.expand_dims(translation_vector, [0])
            translated_centers = centers + tf.expand_dims(translation_vector, [0])
            
            M = -2. * tf.linalg.cross(displacement, efield) * angstrom / electronvolt # Delta dipole contribution
            M -= tf.linalg.cross(my_pos + translation_vector, returning_force) # The original force on the center particle is zero
            
            position_core = tf.reshape(tf.reduce_sum(translated_positions**2, [1]), [-1, 1, 1]) * tf.expand_dims(tf.eye(3, dtype=self.float_type), [0]) - tf.expand_dims(translated_positions, [1]) * tf.expand_dims(translated_positions, [2])
            center_core = tf.reshape(tf.reduce_sum(translated_centers**2, [1]), [-1, 1, 1]) * tf.expand_dims(tf.eye(3, dtype=self.float_type), [0]) - tf.expand_dims(translated_centers, [1]) * tf.expand_dims(translated_centers, [2])
            Q_2 = tf.reduce_sum(position_core * tf.reshape(position_weights, [-1, 1, 1]), [0]) + tf.reduce_sum(center_core * tf.reshape(center_weights, [-1, 1, 1]), [0])
            
            lambd = tf.expand_dims(-returning_force / Q_0, [0])
            mu = tf.expand_dims(tf.linalg.matvec(tf.linalg.inv(Q_2), M), [0])
            
            correction_pos_force = (lambd + tf.linalg.cross(tf.tile(mu, [tf.shape(position_weights)[0], 1]), translated_positions)) * tf.expand_dims(position_weights, [1])
            correction_cen_force = (lambd + tf.linalg.cross(tf.tile(mu, [tf.shape(center_weights)[0], 1]), translated_centers)) * tf.expand_dims(center_weights, [1])
        
        augmented_center_forces = tf.expand_dims(mask, [1]) * tf.expand_dims(returning_force, [0])
        return centers, correction_pos_force, correction_cen_force + augmented_center_forces, delta_E
        
    
    def _do_nothing(self, data):
        return data['centers'], 0., 0., 0.
        
   
    def augment(self, data):
        coin = tf.random.uniform([])
        return tf.cond(coin < self.percentage, lambda: self._augment(data), lambda: self._do_nothing(data))
    

def convert_np_value(value, float_converter):
    if type(value) == int:
        return np.int32(value)
    elif type(value) == float:
        return float_converter(value)
    elif type(value) == bool:
        return np.array(value, dtype=np.bool)
    elif type(value) == np.ndarray:
        if value.dtype == np.int:
            return np.int32(value)
        elif value.dtype == np.float:
            return float_converter(value)
        elif value.dtype == np.bool:
            return value
        else:
            raise RuntimeError('Could not convert the value of dtype %s' % str(value.dtype))
    else:
        try:
            return float_converter(value)
        except:
            raise RuntimeError('Could not convert the value of type %s' % str(type(value)))
            
 
class TFRWriter(object):
    def __init__(self, filename, list_of_properties = ['positions', 'numbers', 'centers', 'energy', 'rvec', 'forces', 'efield'], float_type = 32,
                 verbose=True, reference = 0., filter_centers = False, preprocess_model = None, per_atom_reference = 0.):
        ''' Possible properties:
            positions ('pos' in xyz)
            numbers ('Z' in xyz)
            energy
            rvec ('Lattice' in xyz)
            forces ('force' in xyz)
            vtens
            other
        '''
        if float_type == 32:
            self.float_converter = np.float32
        elif float_type == 64:
            self.float_converter = np.float64
        
        self.filename = filename   
        self.tfr_file = tf.io.TFRecordWriter(self.filename)
        self.list_of_properties = list_of_properties
        self.num_configs = 0
        self.num_atoms = 0
        self.verbose = verbose
        self.stats = []
        self.reference = reference
        self.per_atom_reference = per_atom_reference
        self.filter_centers = filter_centers
        self.preprocess_model = preprocess_model
        
        if type(reference) == str:
            ref_data = np.genfromtxt(os.path.dirname(__file__) + '/ref/reference_' + reference + '.txt')
            self.ref_energies = ref_data[:, 1] / electronvolt # Contains [H+, H2, CH4, NH3, H2O]
            print('Using reference energies stored in reference_%s' % reference)
        
        if self.verbose:
            print('Using float%d' % float_type)
            print('Storing the following properties: ' + str(self.list_of_properties))
        
    def write(self, **kwargs):
        if self.filter_centers:
            kwargs['centers'] = filter_cores(kwargs['centers'], kwargs['positions'], kwargs['numbers'])
            
        if type(self.reference) == str:
            num_H = np.sum(kwargs['numbers'] == 1)
            num_C = np.sum(kwargs['numbers'] == 6)
            num_N = np.sum(kwargs['numbers'] == 7)
            num_O = np.sum(kwargs['numbers'] == 8)
            num_centers = num_C + num_N + num_O + kwargs['centers'].shape[0]
            Q = num_H + 6 * num_C + 7 * num_N + 8 * num_O - 2 * num_centers # The total charge of the system
            ref_energy = num_C * self.ref_energies[2] + num_N * self.ref_energies[3] + num_O * self.ref_energies[4] + \
                         (num_H - 4 * num_C - 3 * num_N - 2 * num_O - Q) / 2. * self.ref_energies[1] + Q * self.ref_energies[0]
        else:
            ref_energy = self.reference
        kwargs['energy'] -= float(ref_energy)
        kwargs['energy'] -= float(self.per_atom_reference * len(kwargs['numbers']))
        
        if not self.preprocess_model is None: # TODO make the input and computed properties more general
            preprocessed_output = self.preprocess_model.compute_static(kwargs['positions'], kwargs['numbers'], kwargs['centers'], 
                                                                       efield = kwargs['efield'], rvec = kwargs['rvec'].reshape([3, 3]), list_of_properties = ['energy', 'forces', 'stress'])
            kwargs['energy'] -= preprocessed_output['energy']
            kwargs['forces'] -= preprocessed_output['forces']
            kwargs['center_forces'] -= preprocessed_output['center_forces']
            kwargs['stress'] -= preprocessed_output['stress']
            kwargs['efield'] = np.zeros([3], dtype=np.float)

        feature = {}
        to_store = self.list_of_properties.copy()
        for key, value in kwargs.items():
            if not key in self.list_of_properties:
                raise RuntimeError('Key %s does not appear in the list of properties' % key)
            to_store.remove(key)

            value = convert_np_value(value, self.float_converter)
            feature[key] = _bytes_feature(tf.compat.as_bytes(value.tostring()))
            
        if len(to_store) >= 1:
            raise RuntimeError('Missing properties: ' + str(to_store))   
        
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        self.tfr_file.write(example.SerializeToString())

        self.num_configs += 1
        self.num_atoms += len(kwargs['numbers'])
        
        if 'energy' in self.list_of_properties:
            self.stats.append(kwargs['energy'])
        
        if self.num_configs % 1000 == 0 and self.verbose:
            print('Storing configuration %d' % self.num_configs) # Can be replaced by a simple progressbar
        
    def write_from_xyz(self, xyzfiles):
        if type(xyzfiles) == str:
            xyzfiles = [xyzfiles]
        
        for filename in xyzfiles:
            for data in load_xyz(filename):
                # Look at list_of_properties and find the corresponding key
                kwargs = {}
                all_numbers = data['Z']
                for item in self.list_of_properties:
                    if item == 'positions':
                        kwargs['positions'] = data['pos'][np.where(all_numbers != 99)]
                    elif item == 'centers':
                        kwargs['centers'] = data['pos'][np.where(all_numbers == 99)]
                    elif item == 'numbers':
                        kwargs['numbers'] = data['Z'][np.where(all_numbers != 99)]
                    elif item == 'rvec':
                        kwargs['rvec'] = data['Lattice']
                    elif item == 'forces':
                        kwargs['forces'] = data['force'][np.where(all_numbers != 99)]
                    else:
                        kwargs[item] = data[item]
                        
                self.write(**kwargs)
    
    def close(self):
        self.tfr_file.close()
        
        if self.verbose:
            print('')
            print('%d configurations were written to file %s' % (self.num_configs, self.filename))
            print('In total, %d atoms were stored' % self.num_atoms)
            
            if 'energy' in self.list_of_properties:
                print('Mean energy: %f' % np.mean(self.stats))
                print('Std energy: %f' % np.std(self.stats))

                print('Max energy: %f' % np.max(self.stats))
                print('Min energy: %f' % np.min(self.stats))
            
            print('')   
