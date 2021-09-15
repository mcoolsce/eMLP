import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from emlp.training import Trainer
from emlp.learning_rate_manager import ExponentialDecayLearningRate
from emlp.schnet import SchNet
from emlp.datasets import DataSet, TFRWriter, DataAugmentation
from emlp.reference import ConsistentFragmentsReference
from emlp.losses import MSE, MAE
from emlp.hooks import SaveHook
from emlp.longrange import LongrangeCoulomb
import tensorflow as tf
from glob import glob

list_of_properties = ['positions', 'numbers', 'centers', 'energy', 'forces', 'efield'] # Properties stored in the tfr-files.

# Switch to True, if the datasets should be generated
if False:
    writer = TFRWriter('validation.tfr', list_of_properties = list_of_properties, reference = 'pbe0_aug-cc-pvtz', filter_centers = True)
    writer.write_from_xyz('validation.xyz')
    writer.close()
    
if False:
    writer = TFRWriter('train.tfr', list_of_properties = list_of_properties, reference = 'pbe0_aug-cc-pvtz', filter_centers = True)
    trainfiles = glob('train_directory/*.xyz')
    writer.write_from_xyz(trainfiles)
    writer.close()

# Choose your strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Remove the DataAugmentation() argument, if no data augmentation is needed.
    train_data = DataSet(['train.tfr'], num_configs = 3090500, cutoff = 4.0, longrange_cutoff = 16.5, batch_size = 64, float_type = 32, num_parallel_calls = 8, 
                         strategy = strategy, list_of_properties = list_of_properties, augment_data = DataAugmentation())
    validation_data = DataSet(['validation.tfr'], num_configs = 5676, cutoff = 4.0, longrange_cutoff = 16.5, batch_size = 64, float_type = 32, num_parallel_calls = 8, 
                              strategy = strategy, list_of_properties = list_of_properties, test = True)
    
    # Starting from anew
    model = SchNet(cutoff = 4., n_max = 32, num_layers = 4, start = 0.0, end = 4.0, num_filters = 128, num_features = 512, shared_W_interactions = False, float_type = 32, 
                   cutoff_transition_width = 0.5, reference = ConsistentFragmentsReference('pbe0_aug-cc-pvtz'), longrange_compute = LongrangeCoulomb())  
    # Starting from an existing model
    #model = SchNet.from_restore_file('resume_name', reference = ConsistentFragmentsReference('pbe0_aug-cc-pvtz'), longrange_compute = LongrangeCoulomb())
      
    optimizer = tf.optimizers.Adam(3e-04)
    learning_rate_manager = ExponentialDecayLearningRate(initial_learning_rate = 3e-04, decay_rate = 0.5, decay_epochs = 30)

    losses = [MSE('energy', scale_factor = 1., per_atom = True), MSE('forces', scale_factor = 1.), MSE('center_forces', scale_factor = 1.)]
    validation_losses = [MAE('energy', per_atom = True), MAE('forces', scale_factor = 1.), MAE('center_forces', scale_factor = 1.)]
         
    savehook = SaveHook(model, ckpt_name = 'model_dir/model_name', max_to_keep = 5, save_period = 1.0, history_period = 8.0,
                        npz_file = 'model_dir/model_name.npz')

    trainer = Trainer(model, losses, train_data, validation_data, strategy = strategy, optimizer = optimizer, savehook = savehook, 
                      learning_rate_manager = learning_rate_manager, validation_losses = validation_losses)
    # Set these to False, when running on the hpc!
    trainer.train(verbose = False, validate_first = False)


