import numpy as np


class LearningRateManager(object):
    def __init__(self):
        pass
        
    def update(self):
        raise NotImplementedError
    
    def initialize(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        
    def decay(self, current_epoch, best_epoch):
        return best_epoch
        
    def stop_training(self):
        return False
        
class ConstantDecayLearningRate(LearningRateManager):
    def __init__(self, initial_learning_rate = 1e-04, decay_factor = 0.5, min_learning_rate = 1e-07, decay_patience = 25):
        super(ConstantDecayLearningRate).__init__()
        
        self.learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate
        self.decay_patience = decay_patience
        
    def update(self):
        return self.learning_rate
        
    def decay(self, current_epoch, best_epoch):
        if current_epoch >= best_epoch + self.decay_patience:
            self.learning_rate *= self.decay_factor
              
            print('The learning rate has been lowered from %.3g to %.3g' % (self.learning_rate / self.decay_factor, self.learning_rate))
            
            return current_epoch
        return best_epoch
          
    def stop_training(self):
        if self.learning_rate < self.min_learning_rate:
            return True
        return False
        
        
class TriangularLearningRate(LearningRateManager):
    def __init__(self, min_lr = 1e-04, max_lr = 1e-03, half_width_in_epochs = 4):
        super(TriangularLearningRate).__init__()
        
        self.learning_rate = min_lr
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.half_width_in_epochs = half_width_in_epochs
        
        self.epoch_progress = 0.
        
    def update(self):
        self.epoch_progress += 1. / self.steps_per_epoch
        
        if self.epoch_progress > 2 * self.half_width_in_epochs:
            self.learning_rate = self.min_lr
            
        elif self.epoch_progress > self.half_width_in_epochs:
            self.learning_rate = self.max_lr + (self.min_lr - self.max_lr) * (self.epoch_progress - self.half_width_in_epochs) / self.half_width_in_epochs
            
        else:
            self.learning_rate = self.min_lr + (self.max_lr - self.min_lr) * self.epoch_progress / self.half_width_in_epochs
            
        return self.learning_rate
        
class ExponentialDecayLearningRate(LearningRateManager):
    def __init__(self, initial_learning_rate = 1e-04, decay_rate = 0.5, decay_epochs = 30):
        super(ExponentialDecayLearningRate).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs
        
        self.epoch_progress = 0.
        
    def update(self):
        self.epoch_progress += 1. / self.steps_per_epoch
        
        self.learning_rate = self.initial_learning_rate * self.decay_rate**(self.epoch_progress / self.decay_epochs)
        return self.learning_rate
