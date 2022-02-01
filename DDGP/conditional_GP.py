# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from DDGP.kernels import *
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

def conditional_GP_Euclidean(Xnew, X, Xnew_mean_function, l, dim_layer, num_layers, q_mu, q_sqrt, 
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



def conditional_GP_Wasserstein(Xnew_mean, Xnew_var, X_mean, X_var, Xnew_mean_function, l, dim_layer, num_layers, q_mu, q_sqrt, 
    log_lengthscales, log_kernel_variance,
    training_time, 
    white=True, full_cov=True):

    print(' ******* conditional GP ****************')

    type_var = 'full'
    num_data = tf.shape(Xnew_mean)[0]  # M

    ##### Wasserstein-2 domain #####

    Kmm_w = wasserstein_2_distance_gaussian_kernel(X1_mean = X_mean, X2_mean = X_mean, X1_var = X_var, X2_var = X_var, log_lengthscale = log_lengthscales,
        log_kernel_variance = log_kernel_variance)
    Kmn_w = wasserstein_2_distance_gaussian_kernel(X1_mean = X_mean, X2_mean = Xnew_mean, X1_var = X_var, X2_var = Xnew_var, log_lengthscale = log_lengthscales,
        log_kernel_variance = log_kernel_variance)
            
    if full_cov:
        Knn_w = wasserstein_2_distance_gaussian_kernel(X1_mean = Xnew_mean, X2_mean = Xnew_mean, X1_var = Xnew_var, X2_var = Xnew_var,
            log_lengthscale = log_lengthscales, log_kernel_variance = log_kernel_variance)
    else:
        Knn_w = RBF_Kdiag(Xnew_mean,log_kernel_variance)

    ##### Euclidean domain ######

    Xnew = Xnew_mean + tf.multiply(tf.sqrt(Xnew_var),
            tf.random.normal(shape=(tf.shape(Xnew_mean)[0], tf.shape(Xnew_mean)[1]), dtype=DTYPE))	
    X = X_mean + tf.multiply(tf.sqrt(X_var),
            tf.random.normal(shape=(tf.shape(X_mean)[0], tf.shape(X_mean)[1]), dtype=DTYPE))	

    Kmn_e = RBF_without_kernel_variance(X, Xnew, log_lengthscales, log_kernel_variance)
    Kmm_e = RBF_without_kernel_variance(X, X, log_lengthscales, log_kernel_variance)
    #Kmm = condition(Kmm)        

    if full_cov:
        Knn_e = RBF_without_kernel_variance(Xnew, Xnew, log_lengthscales, log_kernel_variance)
    else:
        Knn_e = RBF_Kdiag_without_kernel_variance(Xnew, log_kernel_variance)

    ### combine domains ###
    Kmm = Kmm_e * Kmm_w
    Kmn = Kmn_e * Kmn_w
    Knn = Knn_e * Knn_w
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

