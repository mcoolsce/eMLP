import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from emlp.md import NVE, NVT, NPT
from emlp.longrange import LongrangeEwald, LongrangeCoulomb
from emlp.reference import ConstantFragmentsReference
from emlp.schnet import SchNet
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
    # In this example, we will run an NVT simulation on methane with one of the fully trained models on the eQM7 dataset.

    # First, the model is loaded. The flag xla=True can potentially accelerate GPUs simulates at the expensive of more tracing.
    model = SchNet.from_restore_file('models/eQM7_aug1', longrange_compute = LongrangeCoulomb(), reference = ConstantFragmentsReference(float_type = tf.float64), float_type = 64, xla = False)
    
    # Specify your initial configuration. Here, we load the first configuration of the validation set, which is methane
    filename = 'data/validation.xyz'
    positions, numbers, centers, rvec = load_data(filename, filter_centers = True)

    if rvec is None:
        system = System(numbers, positions * angstrom)
    else:
        system = System(numbers, positions * angstrom, rvecs = (rvec * angstrom).astype(np.float))
    
    # Run the MD simulation for 10000 steps, while printing information to the screen every screenprint=10 steps and 
    # storing the output every nprint=1 steps in the output xyz-file
    NVT(system, model, 10000, centers = centers, screenprint = 10, nprint = 1, dt = 0.5, temp = 300,
        name = 'md_run1', efield = [0.0, 0.0, 0.0])
    
