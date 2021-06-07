import tensorflow as tf
from molmod.units import angstrom
from .model import Model
from .reference import ConstantReference
import numpy as np


class LongrangeModel(Model):
    def __init__(self, longrange_compute, float_type = 32):
        Model.__init__(self, cutoff = longrange_compute.cutoff, float_type = float_type, longrange_compute = longrange_compute)
        self.reference = ConstantReference(0.)
        
    
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):
        batches = tf.shape(numbers)[0]
        N = tf.shape(numbers)[1]
        return tf.zeros([batches], dtype = self.float_type), tf.zeros([batches, N], dtype = self.float_type)

     
class LongrangeCoulomb(object):
    def __init__(self, cutoff = 16.5, sigma = 1.2727922061357857):
        self.cutoff = cutoff
        self.sigma = sigma
        print('Direct coulomb interactions with rcut: %f and sigma: %f' % (self.cutoff, sigma))
        
    
    def __call__(self, charges, neighbor_charges, all_positions, dists, rvecs, elements_mask, neighbor_mask, float_type = tf.float32):
        gamma = 1. / (2 * self.sigma)
        A = gamma / np.sqrt(np.pi)
                
        charge_matrix = tf.expand_dims(charges, [2]) * neighbor_charges
        radial_scaling = tf.math.erf(gamma * dists) / (dists + 1e-10) # Avoid zero division

        ''' THE COULOMB ENERGY '''
        coulomb_matrix = charge_matrix * radial_scaling * neighbor_mask * tf.expand_dims(elements_mask, [-1])
        coulomb_energy = tf.reduce_sum(coulomb_matrix, [1, 2]) / 2. / angstrom
        
        ''' THE SELF ENERGY '''
        self_energy = A * tf.reduce_sum(charges**2 * elements_mask, [1]) / angstrom

        return coulomb_energy + self_energy


class LongrangeEwald(object):
    def __init__(self, cutoff = 12., alpha_scale = 3.2, gscale = 1.5, sigma = 1.2727922061357857):
        self.cutoff = cutoff
        self.alpha = alpha_scale / self.cutoff
        self.gcut = gscale * self.alpha
        self.sigma = sigma
        print('Ewald summation with rcut: %f, alpha_scale: %f, gscale: %f, sigma: %f' % (self.cutoff, alpha_scale, gscale, sigma))
        

    def __call__(self, charges, neighbor_charges, all_positions, dists, rvecs, elements_mask, neighbor_mask, float_type = tf.float32):
        # Imposing a smooth cutoff
        smooth_cutoff_mask = f_cutoff(dists, cutoff = self.cutoff, cutoff_transition_width = 0.5, float_type = float_type) * neighbor_mask
    
        ''' The Ewald summation '''
        real_space_energy = real_space_part(self.alpha, charges, neighbor_charges, dists, elements_mask, smooth_cutoff_mask, self.sigma, float_type = float_type)
        self_correction = self_correction_part(self.alpha, charges, elements_mask)
        reciprocal_energy = reciprocal_part(self.alpha, self.gcut, charges, all_positions, rvecs, elements_mask, float_type = float_type)

        ''' THE SELF ENERGY '''
        gamma = 1. / (2 * self.sigma)
        A = gamma / np.sqrt(np.pi)
        self_energy = A * tf.reduce_sum(charges**2 * elements_mask, [1]) / angstrom
        
        return real_space_energy - self_correction + reciprocal_energy + self_energy
        
        
def f_cutoff(input_tensor, cutoff, cutoff_transition_width = 0.5, float_type = tf.float32):
    r_transition = tf.where(input_tensor > cutoff, tf.zeros(tf.shape(input_tensor), dtype = float_type),
                            0.5 * (1 + tf.cos(np.pi * (input_tensor - cutoff + cutoff_transition_width) / cutoff_transition_width)))

    return tf.where(input_tensor > cutoff - cutoff_transition_width, r_transition, tf.ones(tf.shape(input_tensor), dtype = float_type))
        

def real_space_part(alpha, charges, neighbor_charges, dists, elements_mask, radial_mask, sigma, float_type = tf.float32):
    gamma = 1. / (2 * sigma)
    
    charge_matrix = tf.expand_dims(charges, [2]) * neighbor_charges * tf.expand_dims(elements_mask, [-1])
    coulomb_matrix = charge_matrix / (dists + 1e-20) * tf.math.erf(gamma * dists) / angstrom
    
    dist_function = tf.math.erf(alpha * dists * angstrom) / dists / angstrom
    screened_matrix = dist_function * charge_matrix
    
    return 0.5 * tf.reduce_sum((coulomb_matrix - screened_matrix) * radial_mask,  [1, 2])
    
    
def self_correction_part(alpha, charges, elements_mask):
    return alpha / np.sqrt(np.pi) * tf.reduce_sum(charges**2 * elements_mask, [1])
    
    
def generate_kvecs(gvecs, rvecs, gcut, float_type = tf.float32):
    gspacings = 1. / tf.sqrt(tf.reduce_sum(rvecs**2, [1])) # [batches, 3]
    gmax = tf.cast(tf.math.ceil(gcut / (tf.reduce_min(gspacings) / angstrom) - 0.5), tf.int32) # A number, the minimum of all batches

    gx = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [-1, 1, 1, 1]), [1, 2 * gmax + 1, 2 * gmax + 1, 1])
    gy = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [1, -1, 1, 1]), [2 * gmax + 1, 1, 2 * gmax + 1, 1])
    gz = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [1, 1, -1, 1]), [2 * gmax + 1, 2 * gmax + 1, 1, 1])
    
    n = tf.reshape(tf.concat((gx, gy, gz), 3), [(2 * gmax + 1)**3, 3]) # [K, 3]
    k_vecs = 2 * np.pi * tf.einsum('ijk,lk->ilj', gvecs, tf.cast(n, dtype = float_type)) # [batches, K, 3]
    
    # Constructing the mask
    k2 = tf.reduce_sum(k_vecs**2, [2])
    n2 = tf.reduce_sum(n**2, [1])
    
    k_mask_less = tf.cast(tf.less_equal(k2 / angstrom**2, (gcut * 2 * np.pi)**2), dtype = float_type)
    k_mask_zero = tf.cast(tf.not_equal(n2, 0), dtype = float_type)
    
    k_mask = k_mask_less * tf.expand_dims(k_mask_zero, 0)
    
    return k_vecs, k2, k_mask
    
def reciprocal_part(alpha, gcut, charges, positions, rvecs, elements_mask, float_type = tf.float32): 
    gvecs = tf.linalg.inv(rvecs) #  Reciprocal cell matrix
    volume = tf.linalg.det(rvecs) * angstrom**3
    
    k_vecs, k2, k_mask = generate_kvecs(gvecs, rvecs, gcut, float_type = float_type) # [batches, K, 3]

    kr = tf.einsum('ijk,ilk->ijl', k_vecs, positions) # [batches, K, N]
    k2 /= angstrom**2
    
    cos_term = tf.reduce_sum(tf.expand_dims(charges, [1]) * tf.math.cos(kr) * tf.expand_dims(elements_mask, [1]), [2]) # [batches, K]
    sin_term = tf.reduce_sum(tf.expand_dims(charges, [1]) * tf.math.sin(kr) * tf.expand_dims(elements_mask, [1]), [2]) # [batches, K]
    
    rho_k2 = cos_term**2 + sin_term**2
    k_factor = 4 * np.pi / (2 * tf.expand_dims(volume, [1])) * tf.exp(- k2 / (4 * alpha**2)) / (k2 + 1. - k_mask) # Avoid dividing by zero
    
    return tf.reduce_sum(rho_k2 * k_factor * k_mask, [1])

