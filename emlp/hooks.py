import tensorflow as tf
import numpy as np
from glob import glob
import os

      
class SaveHook(object):
    def __init__(self, model, ckpt_name, max_to_keep = 5, save_period = 1., history_period = None, npz_file = None):
        self.ckpt = tf.train.Checkpoint(model = model)
        self.ckpt_name = ckpt_name
        self.max_to_keep = max_to_keep
        self.save_period = save_period
        self.history_period = history_period
        self.npz_file = npz_file
        self.model = model
        
        self.npz_data = {'learning_rate':[], 'total_time':[], 'time_passed':[], 'epoch':[], 
                         'average_training_loss':[]}
        self.npz_keys = []
        
        self.iteration = -1
        
        self.number_list = []
        
        if not os.path.exists(os.path.split(ckpt_name)[0]):
            os.mkdir(os.path.split(ckpt_name)[0])
            
        if not self.history_period is None:
            self.history_dir = os.path.split(ckpt_name)[0] + '/history_' + os.path.split(ckpt_name)[1]
            if not os.path.exists(self.history_dir):
                os.mkdir(self.history_dir)   
            self.history_name = self.history_dir + '/' + os.path.split(ckpt_name)[1]
            
            self.next_history_save = self.history_period
            
            
    def request_steps(self, current_step = 0):
        self.iteration += 1
        steps = int(np.ceil(self.steps_per_cycle * (self.iteration + 1) - current_step))
        return steps

   
    def initialize(self, data_per_epoch, batch_size, validation_losses):
        if self.save_period < batch_size / data_per_epoch:
            raise RuntimeError('The save period %f should be larger than the period of 1 batch size: %f' % (self.save_period, batch_size / data_per_epoch))
        self.data_per_epoch = data_per_epoch
        self.batch_size = batch_size
        self.steps_per_cycle = self.data_per_epoch * self.save_period / self.batch_size
        
        for loss in validation_losses:
            title = loss.get_title()
            self.npz_data[title] = []
            self.npz_keys.append(title)
        self.npz_data['batch_size'] = batch_size
        
        return self.request_steps()
        
        
    def update_npz(self, validation_losses, epoch, average_training_loss, time_passed, total_time, learning_rate):
        for index, title in enumerate(self.npz_keys):
            self.npz_data[title].append(validation_losses[index])
    
        self.npz_data['learning_rate'].append(learning_rate)
        self.npz_data['average_training_loss'].append(average_training_loss)
        self.npz_data['time_passed'].append(time_passed)
        self.npz_data['total_time'].append(total_time)
        self.npz_data['epoch'].append(epoch)
                
        np.savez(self.npz_file, **self.npz_data)
        
    
    def save_history(self, epoch):
        if not self.history_period is None:
            if epoch >= self.next_history_save:
                self.ckpt.write(self.history_name + '_%.2f' % epoch)
                self.model.save(self.history_name + '_%.2f' % epoch)
                self.next_history_save += self.history_period
        
        
    def save(self, epoch):
        self.ckpt.write(self.ckpt_name + '_%.2f' % epoch)
        self.model.save(self.ckpt_name + '_%.2f' % epoch)
        
        self.number_list.append(epoch)
        
        if not self.max_to_keep is None:
            if len(self.number_list) > self.max_to_keep: # Remove the latest one
                remove_index = self.number_list.pop(0)
                for file in glob(self.ckpt_name + '_%.2f.*' % remove_index):
                    os.remove(file)

        
