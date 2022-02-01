# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float32
from DGP.propagate_layers import *
from DGP.model_details import model_details
from DGP.kernels import *
import tensorflow_probability as tfp

class network_architectures(model_details):

    def __init__(self, **kwargs):
        
        model_details.__init__(self, **kwargs)

    def standard_DeepGP(self, X, X_mean_function, training_time, propagate_layers_object):

        if training_time:
            list_KL = []

        ### Euclidean Space ###
        for l in range(1, self.num_layers+1):
      
            
            output_now = propagate_layers_object.propagate_layer(X = X, X_mean_function = X_mean_function, l = l, training_time = training_time, full_cov = False)

            f_mean = output_now[0]
            f_var = output_now[1]+output_now[2]
            if training_time:
                list_KL.append(output_now[3])

            X = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

        if training_time:
            return f_mean, f_var, tf.reduce_sum(list_KL)
        else:
            return f_mean, f_var


    def extract_correlation_rl(self, X_focus, X_expanded, X_focus_mean_function, X_expanded_mean_function, propagate_layers_object):

        training_time = False
        
        ############################################
        ### Extract RL sample for expanded space ###
        ############################################
        list_Knm = []

        for l in range(1, self.num_layers+1):
      

            with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):
                            
                log_kernel_variance = tf.compat.v1.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
                    name='log_kernel_variance')       
                log_lengthscales = tf.compat.v1.get_variable(initializer = tf.constant([0.301 for _ in range(self.dim_layers[l-1])]),
                    dtype=tf.float32,name='log_lengthscales')
            Knm = RBF(X_focus, X_expanded, log_lengthscales, log_kernel_variance)
            list_Knm.append(Knm)

            output_now = propagate_layers_object.propagate_layer(X = X_expanded, X_mean_function = X_expanded_mean_function, l = l, training_time = training_time, full_cov = False)

            f_mean = output_now[0]
            f_var = output_now[1]+output_now[2]

            X_expanded = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	
      
            output_now = propagate_layers_object.propagate_layer(X = X_focus, X_mean_function = X_focus_mean_function, l = l, training_time = training_time, full_cov = False)

            f_mean = output_now[0]
            f_var = output_now[1]+output_now[2]

            X_focus = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

       	
        return list_Knm


    def uncertainty_decomposed_DeepGP(self, X, X_mean_function, propagate_layers_object):

        for l in range(1, self.num_layers+1):
            
            output_now = propagate_layers_object.propagate_layer(X = X, X_mean_function = X_mean_function, l = l, training_time = False, full_cov = False)

            f_mean = output_now[0]
            f_var_epistemic = output_now[1]
            f_var_distributional = output_now[2] 
            f_var = f_var_epistemic + f_var_distributional
            
            X = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

        return f_mean, f_var_epistemic, f_var_distributional


    def uncertainty_decomposed_DeepGP_layer_wise(self, X, X_mean_function, propagate_layers_object):

        print('___________________________________________________________________________________________')

        print('**************** inside uncertainty_decomposed_DeepGP_layer_wise **************************')

        print('___________________________________________________________________________________________')
        list_f_mean = []
        list_f_var_epistemic = []
        list_f_var_distributional = []
        list_X = []

        for l in range(1, self.num_layers+1):
            

            output_now = propagate_layers_object.propagate_layer(X = X, X_mean_function = X_mean_function, l = l, training_time = False, full_cov = False)
            
            f_mean = output_now[0]
            f_var_epistemic = output_now[1]
            f_var_distributional = output_now[2] 
            f_var = f_var_epistemic + f_var_distributional

    
            X = f_mean + tf.multiply(tf.sqrt(f_var),
                tf.random.normal(shape=(tf.shape(f_mean)[0], tf.shape(f_mean)[1]), dtype=DTYPE))	

            list_f_mean.append(f_mean)
            list_f_var_epistemic.append(f_var_epistemic)
            list_f_var_distributional.append(f_var_distributional)
            list_X.append(X)

        return list_X, list_f_mean, list_f_var_epistemic, list_f_var_distributional

    def uncertainty_decomposed_DeepGP_layer_wise_hidden_layers(self, X, X_mean_function, propagate_layers_object):

        list_f_mean = []
        list_f_var_epistemic = []
        list_f_var_distributional = []

        for l in range(1, self.num_layers+1):
            
            output_now = propagate_layers_object.propagate_layer(X = X, X_mean_function = X_mean_function, l = l, training_time = False, full_cov = False)

            f_mean = output_now[0]
            f_var_epistemic = output_now[1]
            f_var_distributional = output_now[2] 
            f_var = f_var_epistemic + f_var_distributional
            
            list_f_mean.append(f_mean)
            list_f_var_epistemic.append(f_var_epistemic)
            list_f_var_distributional.append(f_var_distributional)

        return list_f_mean, list_f_var_epistemic, list_f_var_distributional

    def uncertainty_decomposed_DeepGP_layer_wise_hidden_layers_derivatives(self, X, X_mean_function, propagate_layers_object):

        list_f_mean = []
        list_f_var_epistemic = []
        list_f_var_distributional = []

        for l in range(1, self.num_layers+1):
            
            output_now = propagate_layers_object.propagate_layer_derivatives(X = X, X_mean_function = X_mean_function, l = l, training_time = False)

            f_mean = output_now[0]
            f_var_epistemic = output_now[1]
            f_var_distributional = output_now[2] 
            f_var = f_var_epistemic + f_var_distributional
            
            list_f_mean.append(f_mean)
            list_f_var_epistemic.append(f_var_epistemic)
            list_f_var_distributional.append(f_var_distributional)

        return list_f_mean, list_f_var_epistemic, list_f_var_distributional