import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from emlp.md import NVE, NVT, NPT
from emlp.longrange import LongrangeEwald, LongrangeCoulomb
from emlp.reference import ConstantFragmentsReference
from emlp.schnet import SchNet
from nequip import NEQUIP
from emlp.help_functions import load_xyz, filter_cores
from yaff import *


def load_data(filename, filter_centers=False, verbose = False):
    for data in load_xyz(filename):
        centers = data['pos'][np.where(data['Z'] == 99)]
        positions = data['pos'][np.where(data['Z'] != 99)]
        numbers = data['Z'][np.where(data['Z'] != 99)]
        if 'Lattice' in data.keys():
            rvec = data['Lattice'].reshape([3, 3])
        else:
            rvec = None   
        break
    if filter_centers:
        centers = filter_cores(centers, positions, numbers)
    return positions, numbers, centers, rvec
  
    
if __name__ == '__main__':
    model = SchNet.from_restore_file('model_name', longrange_compute = LongrangeCoulomb(), reference = ConstantFragmentsReference(float_type = tf.float64), float_type = 64)
    filename = 'initial_config_with_centers.xyz'
    positions, numbers, centers, rvec = load_data(filename, filter_centers = True)

    if rvec is None:
        system = System(numbers, positions * angstrom)
    else:
        system = System(numbers, positions * angstrom, rvecs = (rvec * angstrom).astype(np.float))
    
    NVT(system, model, 10000, centers = centers, screenprint = 10, nprint = 1, dt = 0.5, temp = 300,
        name = 'output_name', efield = [0.0, 0.0, 0.0], print_opt_steps = False, xla = False)
    
