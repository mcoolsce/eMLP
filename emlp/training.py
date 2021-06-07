import numpy as np
import tensorflow as tf
import time
from .help_functions import ProgressBar, ScreenLog
    
       
class Trainer(object):
    def __init__(self, model, losses, train_data, validation_data, strategy, savehook, optimizer = tf.optimizers.Adam,
                 learning_rate_manager = None, validation_losses = None):
        self.model = model
        self.strategy = strategy
        self.losses = losses
        if validation_losses is None:
            self.validation_losses = losses
        else:
            self.validation_losses = validation_losses
            
        self.optimizer = optimizer
        self.savehook = savehook
        self.learning_rate_manager = learning_rate_manager
        
        self.train_data = train_data
        self.train_data_iterable = iter(self.train_data._dataset)
        self.validation_data = validation_data
        
        assert self.model.cutoff == self.train_data.cutoff
        assert self.model.cutoff == self.validation_data.cutoff
        if not self.model.longrange_compute is None:
            assert self.model.longrange_compute.cutoff == self.train_data.longrange_cutoff
            assert self.model.longrange_compute.cutoff == self.validation_data.longrange_cutoff
        else:
            print('No longrange interactions are being computed. Use a small longrange cutoff!')
        
        # PRINTING SOME USEFUL INFORMATION
        self.num_gpus = strategy.num_replicas_in_sync
        self.num_gpus_tensor = tf.convert_to_tensor(self.num_gpus, dtype = self.model.float_type)
        print('Number of GPUs: %d' % self.num_gpus)
        self.tv = self.model.trainable_variables
        total_variables = 0
        for variable in self.tv:
            total_variables += np.prod(variable.shape)
        print('Training a model with %d parameters.' % total_variables)

        self.next_training_steps = self.savehook.initialize(data_per_epoch = self.train_data.num_configs, batch_size = self.train_data.batch_size,
                                                            validation_losses = validation_losses)
        
        self.list_of_properties_training = []
        for loss in self.losses:
            for required_property in loss.get_required_properties():
                self.list_of_properties_training.append(required_property)
        self.list_of_properties_validation = []
        for loss in self.validation_losses:
            for required_property in loss.get_required_properties():
                self.list_of_properties_validation.append(required_property)
        #print('Calculated properties when training: %s' % str(self.list_of_properties_training))
        #print('Calculated properties when validating: %s' % str(self.list_of_properties_validation))
        print('')
        
        self.screenlog = ScreenLog(self.validation_losses)      
        self.optimizer.learning_rate = self.learning_rate_manager.learning_rate
        self.learning_rate_manager.initialize(self.train_data.num_configs / self.train_data.batch_size)
        
         
        def compute_loss(losses, output, data): 
            per_gpu_losses = [] 
            for loss in losses:
                sum_losses, sum_elements = loss(output, data) # The loss function will do the required masking
                per_gpu_losses.append((sum_losses, sum_elements))  
            return per_gpu_losses

            
        def single_train_step(**data):
            with tf.GradientTape() as parameter_tape:
                output = model.compute_properties(data, list_of_properties = self.list_of_properties_training + ['masks'])
                per_gpu_losses = compute_loss(self.losses, output, data)
                
                total_loss = 0.
                for sum_losses, num_elements in per_gpu_losses:
                    total_loss += sum_losses / tf.math.maximum(num_elements, 1.)
                
                total_loss /= self.num_gpus_tensor

            parameter_gradients = parameter_tape.gradient(total_loss, self.tv)
            self.optimizer.apply_gradients(zip(parameter_gradients, self.tv))
            
            return total_loss
            
            
        def single_validation_step(**data):
            output = model.compute_properties(data, list_of_properties = self.list_of_properties_validation + ['masks'])
            per_gpu_losses = compute_loss(self.validation_losses, output, data)
            return per_gpu_losses
            
        
        @tf.function(autograph = False, experimental_relax_shapes = True)
        def distributed_validation(data):
            per_gpu_losses = self.strategy.run(single_validation_step, kwargs=data)
            
            batch_losses = []
            for sum_loss, sum_elements in per_gpu_losses:
                sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, sum_loss, axis=None)
                sum_elements = self.strategy.reduce(tf.distribute.ReduceOp.SUM, sum_elements, axis=None) 
                
                batch_losses.append((sum_loss, sum_elements))
                
            return batch_losses
            
            
        @tf.function(autograph = False, experimental_relax_shapes = True)
        def distributed_training(data):   
            total_loss = self.strategy.run(single_train_step, kwargs=data)    
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            
            
        self.distributed_training = distributed_training
        self.distributed_validation = distributed_validation
        
        
    def train(self, verbose = False, validate_first = False):
        progressbar = ProgressBar(self.train_data.num_configs / self.train_data.batch_size, verbose = verbose)
            
        if verbose:
            self.screenlog.print_header()
       
        if validate_first: # Can come in handy if you want to check whether the model is restored well
            total_losses = self.validate_set(self.validation_data._dataset)
            self.screenlog.print_losses(total_losses)
        
        total_time = 0
        best_epoch = 0
        best_mae = np.inf
        epoch = 0
        training_steps = 0
        while True:
            start = time.time()
            
            # In each training cycle, we first process the training dataset
            progressbar.start_training()
            average_training_loss = self.do_training(self.train_data_iterable, self.next_training_steps, progressbar = progressbar)
            
            training_steps += self.next_training_steps
            self.next_training_steps = self.savehook.request_steps(training_steps)
            epoch = training_steps * self.train_data.batch_size / self.train_data.num_configs
            
            # Next we validate
            progressbar.start_validating()
            total_losses = self.validate_set(self.validation_data._dataset, progressbar = progressbar) # A one-shot validation iterable
            total_mae = np.sum(total_losses)
            
            time_passed = time.time() - start
            total_time += time_passed
            
            if verbose:
                self.screenlog.print_losses(total_losses, epoch, average_training_loss.numpy(), time_passed, total_time)
            
            self.savehook.update_npz(total_losses, epoch, average_training_loss.numpy(), time_passed, total_time, self.learning_rate_manager.learning_rate)
            self.savehook.save_history(epoch)
            
            # Saving and early stopping
            if total_mae < best_mae:
                best_mae = total_mae
                best_epoch = epoch
                self.savehook.save(epoch)
                 
            else:
                best_epoch = self.learning_rate_manager.decay(epoch, best_epoch) # Allow the learning rate manager to reset the early stopping timer
            
            # Stop conditions
            if self.learning_rate_manager.stop_training():
                break
            
              
    def do_training(self, dataset, steps, progressbar = None):
        average_loss = 0.
        
        for step in tf.range(steps):
            data = next(dataset)
            average_loss += self.distributed_training(data)
            
            self.optimizer.learning_rate = self.learning_rate_manager.update()
            
            progressbar.step()
            progressbar.draw()
            
        return average_loss / tf.cast(steps, dtype = self.model.float_type)
        
            
    def validate_set(self, dataset, progressbar = None):
        step = 0

        total_losses = np.zeros(len(self.validation_losses))
        num_elements = np.zeros(len(self.validation_losses))
        
        for data in dataset:
            if progressbar is not None:
                progressbar.draw(validating = True)
                
            batch_losses = self.distributed_validation(data)

            for index, loss_data in enumerate(batch_losses):
                total_losses[index] += loss_data[0]
                num_elements[index] += loss_data[1]

        return total_losses / num_elements
        

