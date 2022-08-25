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

# In this example, we will train an eMLP model on the first 10 molecules in the eQM7 dataset (https://doi.org/10.24435/materialscloud:66-9j)
# The reference files are stored in the data folder

# We should let the eMLP know what kind of properties should be read from the data files and internally used in the eMLP
list_of_properties = ['positions', 'numbers', 'centers', 'energy', 'forces', 'efield'] 

# First, the training and validation set are converted to a tensorflow record file (tfr file)
# After the conversion, the total number of configurations being stored in the tfr files
# are printed to the prompt. This number is required for the num_configs argument down below
# in the DataSet class.

# This should only be done once before training. Hence, switch the following two if-statements
# to False, if they have already been generated.

if True:
    writer = TFRWriter('validation.tfr', list_of_properties = list_of_properties, reference = 'pbe0_aug-cc-pvtz', filter_centers = True)
    writer.write_from_xyz('data/validation.xyz')
    writer.close()
    
if True:
    writer = TFRWriter('train.tfr', list_of_properties = list_of_properties, reference = 'pbe0_aug-cc-pvtz', filter_centers = True)
    writer.write_from_xyz('data/train.xyz')
    writer.close()

# Choose your strategy (https://www.tensorflow.org/guide/distributed_training). This becomes important when training on multiple GPUs.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Choose the correct longrange part: LongrangeCoulomb for isolated systems or LongrangeEwald for periodic systems
    longrange_compute = LongrangeCoulomb()

    # Here, we create the train and validation set from the tfr files. The user should specify the number of configurations being stored
    # in those records via the argument num_config. One can remove the DataAugmentation() argument in the training set, if no data augmentation 
    # is needed. The flag test=True is required when initializing the validation set.
    train_data = DataSet(['train.tfr'], num_configs = 4000, cutoff = 4.0, longrange_compute = longrange_compute, batch_size = 64, float_type = 32, num_parallel_calls = 8, 
                         strategy = strategy, list_of_properties = list_of_properties, augment_data = DataAugmentation())
    validation_data = DataSet(['validation.tfr'], num_configs = 1000, cutoff = 4.0, longrange_compute = longrange_compute, batch_size = 64, float_type = 32, num_parallel_calls = 8, 
                              strategy = strategy, list_of_properties = list_of_properties, test = True)
    
    # Here, the SchNet architecture is specified. One can use the ConsistentFragmentsReference('pbe0_aug-cc-pvtz') reference to include the
    # refence structures in every batch to maintain their consistency while training. For all other use cases, just use ConstantReference(value = 0., per_atom = False).
    model = SchNet(cutoff = 4., n_max = 32, num_layers = 4, start = 0.0, end = 4.0, num_filters = 128, num_features = 512, shared_W_interactions = False, float_type = 32, 
                   cutoff_transition_width = 0.5, reference = ConsistentFragmentsReference('pbe0_aug-cc-pvtz'), longrange_compute = longrange_compute)  
    
    # When restarting from a previously trained model, use the line below
    #model = SchNet.from_restore_file('model_dir/model_name_2.00', reference = ConsistentFragmentsReference('pbe0_aug-cc-pvtz'), longrange_compute = LongrangeCoulomb())
    
    # Specify the optimizer and learning rate scheduler
    optimizer = tf.optimizers.Adam(3e-04)
    learning_rate_manager = ExponentialDecayLearningRate(initial_learning_rate = 3e-04, decay_rate = 0.5, decay_epochs = 300)
    
    # Specify your loss function here
    losses = [MSE('energy', scale_factor = 1., per_atom = True), MSE('forces', scale_factor = 1.), MSE('center_forces', scale_factor = 1.)]
    # Specify the validation metrics here
    validation_losses = [MAE('energy', per_atom = True), MAE('forces', scale_factor = 1.), MAE('center_forces', scale_factor = 1.)]
    
    # Specify the save location of the model.     
    savehook = SaveHook(model, ckpt_name = 'model_dir/model_name', max_to_keep = 5, save_period = 1.0, history_period = 8.0,
                        npz_file = 'model_dir/model_name.npz')

    trainer = Trainer(model, losses, train_data, validation_data, strategy = strategy, optimizer = optimizer, savehook = savehook, 
                      learning_rate_manager = learning_rate_manager, validation_losses = validation_losses)
    
    # Set these to False, when running on the hpc to supress the printed output
    trainer.train(verbose = True, validate_first = True)


