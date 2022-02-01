# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
DTYPE=tf.float32

      

def RBF(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / tf.exp(log_lengthscales)
	X2 = X2 / tf.exp(log_lengthscales)
	X1s = tf.reduce_sum(tf.square(X1),1)
	X2s = tf.reduce_sum(tf.square(X2),1)       

	return tf.exp(log_kernel_variance) * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1))) /2)      

def RBF_Kdiag(X, log_kernel_variance):
	### returns a list
	return tf.ones((tf.shape(X)[0],1),dtype=tf.float32) * tf.exp(log_kernel_variance)	


def RBF_without_kernel_variance(X1, X2, log_lengthscales, log_kernel_variance):
       	
	X1 = X1 / tf.exp(log_lengthscales)
	X2 = X2 / tf.exp(log_lengthscales)
	X1s = tf.reduce_sum(tf.square(X1),1)
	X2s = tf.reduce_sum(tf.square(X2),1)       

	return tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1))) /2)      

def RBF_Kdiag_without_kernel_variance(X, log_kernel_variance):
	### returns a list
	return tf.ones((tf.shape(X)[0],1),dtype=tf.float32) 


def RBF_first_derivative_X1(X1, X2, log_lengthscales, log_kernel_variance, dim):

    num_1 = tf.shape(X1)[0]
    num_2 = tf.shape(X2)[0]

    ### X1 -- (num_1, num_dim) ###
    ### X2-- (num_2, num_dim) ###
    print('***** inside RBF_first_derivative')

    current_log_lengthscale = tf.slice(log_lengthscales, [dim], [1])

    typical_se_kernel =  RBF(X1 = X1, X2 = X2, log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance)

    X1_current = tf.tile(tf.slice(X1, [0,dim], [-1,1]), [1, num_2])
    X2_current = tf.tile(tf.transpose(tf.slice(X2, [0,dim], [-1,1])), [num_1, 1])
    
    return tf.multiply(typical_se_kernel, (X2_current - X1_current)) / tf.square(tf.exp(current_log_lengthscale))

def RBF_first_derivative_X2(X1, X2, log_lengthscales, log_kernel_variance, dim):

    num_1 = tf.shape(X1)[0]
    num_2 = tf.shape(X2)[0]

    ### X1 -- (num_1, num_dim) ###
    ### X2-- (num_2, num_dim) ###
    print('***** inside RBF_first_derivative')

    current_log_lengthscale = tf.slice(log_lengthscales, [dim], [1])

    typical_se_kernel =  RBF(X1 = X1, X2 = X2, log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance)

    X1_current = tf.tile(tf.slice(X1, [0,dim], [-1,1]), [1, num_2])
    X2_current = tf.tile(tf.transpose(tf.slice(X2, [0,dim], [-1,1])), [num_1, 1])
    
    return tf.multiply(typical_se_kernel, (X1_current - X2_current)) / tf.square(tf.exp(current_log_lengthscale))


def RBF_second_derivative(X1, X2, log_lengthscales, log_kernel_variance, dim):

    num_1 = tf.shape(X1)[0]
    num_2 = tf.shape(X2)[0]

    ### X1 -- (num_1, num_dim) ###
    ### X2-- (num_2, num_dim) ###
    print('***** inside RBF_second_derivative')

    current_log_lengthscale = tf.slice(log_lengthscales, [dim], [1])

    typical_se_kernel =  RBF(X1 = X1, X2 = X2, log_lengthscales = log_lengthscales, log_kernel_variance = log_kernel_variance)

    X1_current = tf.tile(tf.slice(X1, [0,dim], [-1,1]), [1, num_2])
    X2_current = tf.tile(tf.transpose(tf.slice(X2, [0,dim], [-1,1])), [num_1, 1])
    
    plm = (tf.square(tf.exp(current_log_lengthscale)) - tf.square(X1_current - X2_current))  /  tf.square(tf.square(tf.exp(current_log_lengthscale)))

    return tf.multiply(typical_se_kernel, plm)



def RBF_Kdiag_second_derivative(X, log_kernel_variance, log_lengthscales):
	### returns a list
	return tf.ones((tf.shape(X)[0],1),dtype=tf.float32) * tf.exp(log_kernel_variance)	/ tf.square(tf.exp(log_lengthscales))
