# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float32
from DGP.conditional_GP import *
from DGP.model_details import model_details

### helper functions ###

def condition(X):
    return X + tf.eye(tf.shape(X)[0]) * 1e-2

class propagate_layers(model_details):

    def __init__(self, **kwargs):
        
        model_details.__init__(self, **kwargs)

    def propagate_layer(self, X, X_mean_function, l, training_time, full_cov):

        type_var = 'full'

        with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):
                        
            log_kernel_variance = tf.compat.v1.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
                name='log_kernel_variance')       
            log_lengthscales = tf.compat.v1.get_variable(initializer = tf.constant([0.301 for _ in range(self.dim_layers[l-1])]),
                dtype=tf.float32,name='log_lengthscales')

            if l==1:
                Z = tf.compat.v1.get_variable(initializer =  tf.constant(self.Z_init, dtype=DTYPE),
                    dtype=DTYPE, name='Z')
            else:
                Z = tf.compat.v1.get_variable(initializer =  tf.random_uniform_initializer(minval=-0.5,
                    maxval=0.5), shape = (self.num_inducing[l-1], self.dim_layers[l-1]),
                    dtype=DTYPE,name='Z')

            q_mu = tf.compat.v1.get_variable(initializer = tf.random_uniform_initializer(minval=-0.5,
                maxval=0.5), shape=(self.num_inducing[l-1], self.dim_layers[l]),
                dtype=DTYPE, name='q_mu')
         
            if l!=self.num_layers:
                q_identity_matrix = np.tile(np.expand_dims(1e-1*np.eye(self.num_inducing[l-1], dtype=np.float32),axis=0),(self.dim_layers[l], 1, 1))
            else:
                q_identity_matrix = np.tile(np.expand_dims(np.eye(self.num_inducing[l-1], dtype=np.float32),axis=0),(self.dim_layers[l], 1, 1))
            
            q_cholesky_unmasked = tf.compat.v1.get_variable(initializer = tf.constant(q_identity_matrix, dtype=tf.float32),
                dtype=DTYPE, name='q_cholesky_unmasked')
            print(q_cholesky_unmasked)
            q_sqrt = tf.linalg.band_part(q_cholesky_unmasked,-1,0)
            print(q_sqrt)
        
        output_now = conditional_GP(Xnew = X, X = Z, Xnew_mean_function = X_mean_function, 
            l = l, dim_layer = self.dim_layers[l], num_layers = self.num_layers, q_mu = q_mu, q_sqrt = q_sqrt, 
            log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance,               
            training_time = training_time, white = True, full_cov = full_cov)

        output_mean = output_now[0]
        output_var_epistemic = output_now[1]
        output_var_distributional = output_now[2]
        if training_time:
            kl_term = output_now[3]

        if training_time:
            return output_mean, output_var_epistemic, output_var_distributional, kl_term
        else:
            return output_mean, output_var_epistemic, output_var_distributional

    def propagate_layer_derivatives(self, X, X_mean_function, l, training_time):

        type_var = 'full'
        full_cov = False

        with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):
                        
            log_kernel_variance = tf.compat.v1.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
                name='log_kernel_variance')       
            log_lengthscales = tf.compat.v1.get_variable(initializer = tf.constant([0.301 for _ in range(self.dim_layers[l-1])]),
                dtype=tf.float32,name='log_lengthscales')

            if l==1:
                Z = tf.compat.v1.get_variable(initializer =  tf.constant(self.Z_init, dtype=DTYPE),
                    dtype=DTYPE, name='Z')
            else:
                Z = tf.compat.v1.get_variable(initializer =  tf.random_uniform_initializer(minval=-0.5,
                    maxval=0.5), shape = (self.num_inducing[l-1], self.dim_layers[l-1]),
                    dtype=DTYPE,name='Z')

            q_mu = tf.compat.v1.get_variable(initializer = tf.random_uniform_initializer(minval=-0.5,
                maxval=0.5), shape=(self.num_inducing[l-1], self.dim_layers[l]),
                dtype=DTYPE, name='q_mu')
         
            if l!=self.num_layers:
                q_identity_matrix = np.tile(np.expand_dims(1e-1 *np.eye(self.num_inducing[l-1], dtype=np.float32),axis=0),(self.dim_layers[l], 1, 1))
            else:
                q_identity_matrix = np.tile(np.expand_dims(np.eye(self.num_inducing[l-1], dtype=np.float32),axis=0),(self.dim_layers[l], 1, 1))
            
            q_cholesky_unmasked = tf.compat.v1.get_variable(initializer = tf.constant(q_identity_matrix, dtype=tf.float32),
                dtype=DTYPE, name='q_cholesky_unmasked')
            print(q_cholesky_unmasked)
            q_sqrt = tf.linalg.band_part(q_cholesky_unmasked,-1,0)
            print(q_sqrt)
        
        output_now = conditional_GP_Derivatives(Xnew = X, X = Z, Xnew_mean_function = X_mean_function, 
            l = l, dim_layer = self.dim_layers[l], num_layers = self.num_layers, q_mu = q_mu, q_sqrt = q_sqrt, 
            log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance,               
            training_time = False, white = True, full_cov = full_cov)

        output_mean = output_now[0]
        output_var_epistemic = output_now[1]
        output_var_distributional = output_now[2]


        return output_mean, output_var_epistemic, output_var_distributional
