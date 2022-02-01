# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from DGP.kernels import *
DTYPE=tf.float32


def KL(q_mu, q_sqrt):
    ###############################################
    #### this only works in the white-scenario ####
    ###############################################

    type_var = 'full'
    KL_term = tf.constant(0.0,dtype=DTYPE)
    
    KL_term += - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_sqrt)))
    if type_var=='full':
        KL_term += tf.reduce_sum(tf.linalg.trace(tf.linalg.matmul(q_sqrt,q_sqrt,transpose_b=True)))
    else:
        KL_term += tf.reduce_sum(tf.linalg.trace(tf.square(q_sqrt)))
    
    KL_term += tf.reduce_sum(tf.linalg.matmul(q_mu,q_mu ,transpose_a=True)) - tf.cast(tf.shape(q_mu)[0],DTYPE ) * tf.cast(tf.shape(q_mu)[1],DTYPE )

    return 0.5 * KL_term

def condition(X):

    return X + tf.eye(tf.shape(X)[0]) * 1e-2

def conditional_GP(Xnew, X, Xnew_mean_function, l, dim_layer, num_layers, q_mu, q_sqrt, 
    log_lengthscales, log_kernel_variance,
    training_time, 
    white=True, full_cov=True):

    print(' ******* conditional GP ****************')

    type_var = 'full'
    num_data = tf.shape(Xnew)[0]  # M

    Kmn = RBF(X, Xnew, log_lengthscales, log_kernel_variance)
    Kmm = RBF(X, X, log_lengthscales, log_kernel_variance)
    Kmm = condition(Kmm)        

    Lm = tf.linalg.cholesky(Kmm)
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    inv_lm_t_A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)
    print(inv_lm_t_A)


    ### Compute Epistemic Mean ###
    if l == num_layers:
        fmean = tf.matmul(inv_lm_t_A, q_mu, transpose_a = True) 
    else:
        fmean = tf.matmul(inv_lm_t_A, q_mu, transpose_a = True) + Xnew_mean_function 
    
    if full_cov:
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        fvar_epistemic = tf.matmul(LTA,LTA,transpose_a=True)
    else:
        A = tf.tile(tf.expand_dims(A, axis=0),[ dim_layer,1,1])
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        print('****')
        print(LTA)
        fvar_epistemic = tf.transpose(tf.reduce_sum(tf.square(LTA),1,keepdims=False))


    if full_cov:
        Knn = RBF(Xnew, Xnew, log_lengthscales, log_kernel_variance)
    else:
        Knn = RBF_Kdiag(Xnew, log_kernel_variance)
    
    Lm = tf.linalg.cholesky(Kmm)
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    A = tf.tile(tf.expand_dims(A, axis=0),[ dim_layer,1,1])

    if full_cov:
        fvar_distributional = Knn - tf.matmul(A, A, transpose_a=True)
    else:
        print('******')
        print(A)
        fvar_distributional = Knn - tf.transpose(tf.reduce_sum(tf.square(A), 1,keepdims=False))

    #fvar = fvar_epistemic + fvar_distributional

    if training_time:

        kl_term = KL(q_mu, q_sqrt)

    print('************************** final stuff from conditional GP *******************')
    print(fmean)
    print(fvar_epistemic)
    print(fvar_distributional)


    if training_time:
        return fmean, fvar_epistemic, fvar_distributional, kl_term
    else:
        return fmean, fvar_epistemic, fvar_distributional


def conditional_GP_Derivatives(Xnew, X, Xnew_mean_function, l, dim_layer, num_layers, q_mu, q_sqrt, 
    log_lengthscales, log_kernel_variance,
    training_time, 
    white=True, full_cov=True):

    #################################################
    #### Based on 

    print('******* conditional GP Derivatives ****************')

    type_var = 'full'
    num_data = tf.shape(Xnew)[0]  # M

    #Knm = RBF_first_derivative_X1(Xnew, X, log_lengthscales, log_kernel_variance, dim = 0)
    Kmn = RBF_first_derivative_X2(X, Xnew, log_lengthscales, log_kernel_variance, dim = 0)
    Kmm = RBF(X, X, log_lengthscales, log_kernel_variance)
    Kmm = condition(Kmm)        

    Lm = tf.linalg.cholesky(Kmm)
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    inv_lm_t_A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)
    print(inv_lm_t_A)


    ### Compute Epistemic Mean ###
    if l == num_layers:
        fmean = tf.matmul(inv_lm_t_A, q_mu, transpose_a = True) 
    else:
        fmean = tf.matmul(inv_lm_t_A, q_mu, transpose_a = True) + Xnew_mean_function 
    
    if full_cov:
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        fvar_epistemic = tf.matmul(LTA,LTA,transpose_a=True)
    else:
        A = tf.tile(tf.expand_dims(A, axis=0),[ dim_layer,1,1])
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        print('****')
        print(LTA)
        fvar_epistemic = tf.transpose(tf.reduce_sum(tf.square(LTA),1,keepdims=False))


    if full_cov:
        Knn = RBF_second_derivative(Xnew, Xnew, log_lengthscales, log_kernel_variance, 0)
    else:
        Knn = RBF_Kdiag_second_derivative(Xnew, log_kernel_variance, log_lengthscales)
    
    Lm = tf.linalg.cholesky(Kmm)
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    A = tf.tile(tf.expand_dims(A, axis=0),[ dim_layer,1,1])

    if full_cov:
        fvar_distributional = Knn - tf.matmul(A, A, transpose_a=True)
    else:
        print('******')
        print(A)
        fvar_distributional = Knn - tf.transpose(tf.reduce_sum(tf.square(A), 1,keepdims=False))

    #fvar = fvar_epistemic + fvar_distributional

    if training_time:

        kl_term = KL(q_mu, q_sqrt)

    print('************************** final stuff from conditional GP *******************')
    print(fmean)
    print(fvar_epistemic)
    print(fvar_distributional)
    

    if training_time:
        return fmean, fvar_epistemic, fvar_distributional, kl_term
    else:
        return fmean, fvar_epistemic, fvar_distributional
