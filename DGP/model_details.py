from matplotlib import use
import tensorflow as tf
import numpy as np
from collections import defaultdict


class model_details(object):




    def __init__(self, num_data, dim_input, dim_output,
        num_iterations, num_inducing,
        type_var, num_layers, dim_layers,
        num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type):
        
        self.num_data = num_data
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_iterations = num_iterations
        self.num_inducing = num_inducing
        self.type_var = type_var
        self.num_layers = num_layers
        self.dim_layers = dim_layers
        self.num_batch = num_batch 
        self.Z_init = Z_init 
        self.num_test = num_test 
        self.learning_rate = learning_rate
        self.base_seed = base_seed 
        self.mean_Y_training = mean_Y_training 
        self.dataset_name = dataset_name
        self.use_diagnostics = use_diagnostics
        self.task_type = task_type