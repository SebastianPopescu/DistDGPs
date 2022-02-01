# -*- coding: utf-8 -*-
from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import numpy as np
from collections import defaultdict
import random
import argparse
import matplotlib.pyplot as plt
import sys
import os
DTYPE=tf.float32
import seaborn as sns
from sklearn.cluster import  KMeans
from matplotlib import rcParams
import itertools
from scipy.stats import norm
import pandas as pd
import scipy
from DDGP.model_details import model_details
from DDGP.propagate_layers import *
from DDGP.losses import *
from DDGP.network_architectures import *
sys.setrecursionlimit(10000)

### helper functions ###

def draw_gaussian_at(support, sd=1.0, height=1.0, xpos=0.0, ypos=0.0, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    gaussian = np.exp((-support ** 2.0) / (2 * sd ** 2.0))
    gaussian /= gaussian.max()
    gaussian *= height
    return ax.plot(gaussian + xpos, support + ypos, **kwargs)

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def inv_probit_np(x):
    
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

def find_weights(input_dim, output_dim, X):
    
    """
    Find the initial weights of the Linear mean function based on
    input and output dimensions of the layer
    """

    if input_dim == output_dim:
        W = np.eye(input_dim)

    elif input_dim > output_dim:

        _, _, V = np.linalg.svd(X, full_matrices=False)
        W = V[:output_dim, :].T

    elif input_dim < output_dim:
        I = np.eye(input_dim)
        zeros = np.zeros((input_dim, output_dim - input_dim))
        W = np.concatenate([I, zeros], 1)

    W = W.astype(np.float32)

    return W

def create_objects(num_data, dim_input, dim_output,
        num_iterations, num_inducing,
        type_var, num_layers, dim_layers,
        num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type):

    ### Create objects ###
    propagate_layers_object = propagate_layers(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output,
        num_iterations = num_iterations, 
        num_inducing = num_inducing,
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init, 
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type
        )

    network_architectures_object = network_architectures(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output,
        num_iterations = num_iterations, 
        num_inducing = num_inducing,
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init, 
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type  
        )
    
    cost_functions_object = cost_functions(num_data = num_data, 
        dim_input = dim_input, 
        dim_output = dim_output,
        num_iterations = num_iterations, 
        num_inducing = num_inducing,
        type_var = type_var, 
        num_layers = num_layers, 
        dim_layers = dim_layers,
        num_batch = num_batch, 
        Z_init = Z_init, 
        num_test = num_test, 
        learning_rate = learning_rate, 
        base_seed = base_seed, 
        mean_Y_training = mean_Y_training, 
        dataset_name = dataset_name,
        use_diagnostics = use_diagnostics, 
        task_type = task_type
        )
    


    return network_architectures_object, cost_functions_object, propagate_layers_object


@tf.function
def train_step(X_train_batch, X_train_mean_function_batch, Y_train_batch, network_architectures_object, propagate_layers_object, cost_functions_object, optimizer, task_type):

    with tf.GradientTape() as tape:

        output_training = network_architectures_object.standard_DeepGP(X = X_train_batch, 
            X_mean_function = X_train_mean_function_batch, training_time = True,
            propagate_layers_object = propagate_layers_object)

        if task_type=='regression':
            data_fit_cost, kl_cost = cost_functions_object.regression(inputul = output_training, Y = Y_train_batch)
        elif task_type =='classification':
            data_fit_cost, kl_cost = cost_functions_object.classification(inputul = output_training, Y = Y_train_batch)

        cost = - data_fit_cost + kl_cost

    var = tape.watched_variables()
    print('_________________________we are watching the following variables____________________')
    print(var)
    #with writer.as_default():
    #    # other model code would go here
    #    tf.summary.scalar("my_metric", 0.5, step=step)

    gradients = tape.gradient(cost, var)
    optimizer.apply_gradients(zip(gradients, var))
    data_fit_cost = data_fit_cost / 10.

    ##### TODO -- get hyperparameters #####
    ##### store them in a dictionary ######
    list_Z = []


    for l in range(1, num_layers + 1):

        with tf.compat.v1.variable_scope('num_layer_'+str(l), reuse=tf.compat.v1.AUTO_REUSE):
                        
            #log_kernel_variance = tf.compat.v1.get_variable(initializer = tf.constant(0.301),dtype=tf.float32,
            #    name='log_kernel_variance')       
            #log_lengthscales = tf.compat.v1.get_variable(initializer = tf.constant([0.301 for _ in range(self.dim_layers[l-1])]),
            #    dtype=tf.float32,name='log_lengthscales')

            if l==1:
                Z = tf.compat.v1.get_variable(
                    dtype=DTYPE, name='Z')
            else:
                Z = tf.compat.v1.get_variable(
                    dtype=DTYPE, name='Z_mean')     

            list_Z.append(Z)
            
    with tf.compat.v1.variable_scope('gaussian_likelihood', reuse=tf.compat.v1.AUTO_REUSE):
    
        log_variance_output = tf.compat.v1.get_variable(initializer = tf.constant(-1.0), dtype=tf.float32,
            name='log_variance_output')

    return data_fit_cost, kl_cost, output_training, log_variance_output, list_Z

@tf.function
def test_step(X_test_batch, X_test_mean_function_batch, Y_test_batch, network_architectures_object, propagate_layers_object,  task_type, dim_output, mean_Y_training):

    f_mean_testing, f_var_epistemic_testing, f_var_distributional_testing = network_architectures_object.uncertainty_decomposed_DeepGP(X = X_test_batch, 
        X_mean_function = X_test_mean_function_batch, propagate_layers_object = propagate_layers_object)

    f_var_testing = f_var_epistemic_testing + f_var_distributional_testing

    if task_type=='regression':

        f_mean_testing += mean_Y_training

        ########### MAE on Testing data #################
        mae_testing = tf.reduce_mean(tf.abs(Y_test_batch - f_mean_testing))
        ########### Log-likelihood on Testing Data ######

        with tf.compat.v1.variable_scope('gaussian_likelihood', reuse=tf.compat.v1.AUTO_REUSE):        
            log_variance_output = tf.compat.v1.get_variable(initializer = tf.constant(-1.0),dtype=tf.float32,
                name='log_variance_output')

        nll_test = tf.reduce_sum(variational_expectations(Y_test_batch, f_mean_testing, f_var_testing, log_variance_output)) 

    elif task_type=='classification':

        if dim_output==1:
            f_mean_testing_squashed = inv_probit(f_mean_testing)
    
        else:
            pass

        #########################################
        ##### Metrics for Accuracy ##############
        #########################################

        if dim_output>1:                

            correct_pred_testing = tf.equal(tf.argmax(f_mean_testing,1), tf.argmax(Y_test_batch,1))
            accuracy_testing = tf.reduce_mean(tf.cast(correct_pred_testing, DTYPE))
        else:

            correct_pred_testing = tf.equal(tf.round(f_mean_testing_squashed), Y_test_batch)
            accuracy_testing = tf.reduce_mean(tf.cast(correct_pred_testing, DTYPE))

        #################################################
        ########### Log-likelihood on Testing Data ######
        #################################################
    
        #### we sample 5 times and average ###

        sampled_testing = tf.tile(tf.expand_dims(f_mean_testing, axis=-1), [1,1,5]) + tf.multiply(tf.tile(tf.expand_dims(tf.sqrt(f_var_testing),axis=-1),[1,1,5]),
            tf.random.normal(shape=(tf.shape(f_mean_testing)[0],tf.shape(f_mean_testing)[1],5), dtype=DTYPE))	
        sampled_testing = tf.reduce_mean(sampled_testing, axis=-1, keepdims=False)

        if dim_output == 1:

            ##### Binary classification #####
            nll_test = tf.reduce_sum(bernoulli(p = sampled_testing, x = Y_test_batch))	

        else:

            ###### Multi-class Classification ######
            nll_test = tf.reduce_sum(multiclass_helper(inputul = sampled_testing, outputul = Y_test_batch))

    if task_type=='regression':
         
        return nll_test, mae_testing

    elif task_type=='classification':

        return nll_test, accuracy_testing

@tf.function
def get_predictions(X_test_batch,  X_test_mean_function_batch, network_architectures_object, propagate_layers_object):

    list_samples, list_mean_epistemic, list_var_epistemic, list_var_distributional = network_architectures_object.uncertainty_decomposed_DeepGP_layer_wise(X = X_test_batch, 
        X_mean_function = X_test_mean_function_batch, propagate_layers_object = propagate_layers_object)

    return list_mean_epistemic, list_var_epistemic, list_var_distributional

@tf.function
def get_correlation_rl(X_focus,  X_focus_mean_function, X_expanded, X_expanded_mean_function, network_architectures_object, propagate_layers_object):

    correlation_rl  = network_architectures_object.extract_correlation_rl(X_focus, X_expanded, X_focus_mean_function, X_expanded_mean_function, propagate_layers_object)

    return correlation_rl

#### main function ######
def main_DeepGP( num_data, dim_input, dim_output,
        num_iterations, num_inducing, 
        type_var, num_layers, dim_layers,
        num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
        use_diagnostics, task_type, X_training, Y_training, X_testing, Y_testing):
  
        tf.random.set_seed(base_seed)

        #X_train = tf.placeholder(tf.float32,shape=(None, dim_input), name='X_train')	
        #Y_train = tf.placeholder(tf.float32,shape=(None, dim_output), name='Y_train')

        #X_test = tf.placeholder(tf.float32,shape=(None, dim_input), name='X_test')	
        #Y_test = tf.placeholder(tf.float32,shape=(None, dim_output), name='Y_test')

        W_global = find_weights(input_dim = dim_layers[0] , output_dim = dim_layers[1], X = X_training)

        #### Mean functions for Training and Testing Set #####

        X_training_mean_function = tf.matmul(X_training, W_global)
        X_testing_mean_function = tf.matmul(X_testing, W_global)

        ### Create objects ###
        network_architectures_object, cost_functions_object, propagate_layers_object = create_objects(num_data, dim_input, dim_output,
            num_iterations, num_inducing,
            type_var, num_layers, dim_layers,
            num_batch, Z_init, num_test, learning_rate, base_seed, mean_Y_training, dataset_name,
            use_diagnostics, task_type)
    
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
        where_to_save = str(dataset_name)+'/num_inducing_'+str(num_inducing[-1])+'/lr_'+str(learning_rate)+'/num_layers_'+str(num_layers)+'_dim_layers_'+str(dim_layers[1])+'/seed_'+str(base_seed)


        for i in range(num_iterations):

            #np.random.seed(i+base_seed)
            #### get mini_batch for training data ####
            #lista = np.arange(num_data)
            #np.random.shuffle(lista)
            #current_index = lista[:num_batch]

            data_fit_cost, kl_cost, output_training, log_variance_output, Z = train_step(X_training, X_training_mean_function, Y_training, 
                network_architectures_object, propagate_layers_object, 
                cost_functions_object, opt, task_type)

            data_fit_cost_np = data_fit_cost.numpy()
            kl_cost_np = kl_cost.numpy()
            ### TODO -- output_training -- what does it contain?? ###
            costul_actual = data_fit_cost_np - kl_cost_np

            if i % 250 == 0:

                #### get mini_batch for testing data ####
                #lista = np.arange(num_test)
                #np.random.shuffle(lista)
                #current_index_testing = lista[:num_batch]

                nll_test_now, precision_now = test_step(X_testing, X_testing_mean_function, Y_testing, network_architectures_object, propagate_layers_object, 
                    task_type, dim_output, mean_Y_training)
                print(nll_test_now)
                print(precision_now)
                print('----------')
                total_nll_np=nll_test_now
                precision_testing_overall_np = precision_now.numpy()
                total_nll_np = total_nll_np.numpy() / num_test

                print('at iteration '+str(i) + ' we have nll : '+str(costul_actual) + 're cost :'+str(data_fit_cost_np)+' kl cost :'+str(kl_cost_np) +' precision_testing : '+str(precision_testing_overall_np))	

            if i % 1000==0:

                cmd = 'mkdir -p ./figures/'+str(where_to_save)
                os.system(cmd)
      
                #########################################
                #### produce the Uncertainity Plots #####
                #########################################

                #expanded_space = np.linspace(-4.0, 7.5, 500).reshape((-1,1))

                xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
                grid = np.c_[xx.ravel(), yy.ravel()]
                grid = grid.astype(np.float32)
                grid_mean_function = tf.matmul(grid, W_global)

                #################################################################
                ###### Get predictive mean and variance at hidden layers ########
                #################################################################

                f_mean_overall = defaultdict()
                f_var_overall = defaultdict()
                for current_layer in range(num_layers):
                    f_mean_overall[current_layer] = []
                    f_var_overall[current_layer] = []

                for nvm in range(100):

                    f_mean_epistemic_testing_np, f_var_epistemic_testing_np, f_var_distributional_testing_np = get_predictions(grid, grid_mean_function, network_architectures_object, 
                        propagate_layers_object)

                    for current_layer in range(num_layers):
                        f_mean_overall[current_layer].append(f_mean_epistemic_testing_np[current_layer])
                        f_var_overall[current_layer].append(f_var_distributional_testing_np[current_layer])

                for current_layer in range(num_layers):

                    f_mean_overall[current_layer] = tf.concat(f_mean_overall[current_layer], axis = 1)
                    f_var_overall[current_layer] = tf.concat(f_var_overall[current_layer], axis = 1)

                    f_mean_overall[current_layer] = tf.reduce_mean(f_mean_overall[current_layer], axis = 1)
                    f_var_overall[current_layer] = tf.reduce_mean(f_var_overall[current_layer], axis = 1)

                Z_np = [Z[nvm].numpy() for nvm in range(len(Z))]


                ################################################################################
                ########### Plot the moments and Z moments #####################################
                ################################################################################

                f_mean_overall_inliers = defaultdict()
                f_var_epistemic_overall_inliers = defaultdict()
                f_var_distributional_overall_inliers = defaultdict()

                xx, yy = np.mgrid[-2.:2.:.1, -2.:2.:.1]
                expanded_space_inliers = np.c_[xx.ravel(), yy.ravel()]
                expanded_space_inliers = expanded_space_inliers.astype(np.float32)
                expanded_space_inliers_mean_function = tf.matmul(expanded_space_inliers, W_global)

                f_mean_overall_outliers = defaultdict()
                f_var_epistemic_overall_outliers = defaultdict()
                f_var_distributional_overall_outliers = defaultdict()

                xx, yy = np.mgrid[-5.:-3.:.1, -5.:-3.:.1]
                expanded_space_outliers = np.c_[xx.ravel(), yy.ravel()]
                expanded_space_outliers = expanded_space_outliers.astype(np.float32)
                expanded_space_outliers_mean_function = tf.matmul(expanded_space_outliers, W_global)

                for current_layer in range(num_layers):

                    f_mean_epistemic_testing_np, f_var_epistemic_testing_np, f_var_distributional_testing_np = get_predictions(expanded_space_inliers, expanded_space_inliers_mean_function, 
                        network_architectures_object, propagate_layers_object)

                    f_mean_overall_inliers[current_layer] = f_mean_epistemic_testing_np[current_layer].numpy().ravel()
                    f_var_epistemic_overall_inliers[current_layer] = f_var_epistemic_testing_np[current_layer].numpy().ravel()                
                    f_var_distributional_overall_inliers[current_layer] = f_var_distributional_testing_np[current_layer].numpy().ravel()


                    f_mean_epistemic_testing_np, f_var_epistemic_testing_np, f_var_distributional_testing_np = get_predictions(expanded_space_outliers, expanded_space_outliers_mean_function, 
                        network_architectures_object, propagate_layers_object)

                    f_mean_overall_outliers[current_layer] = f_mean_epistemic_testing_np[current_layer].numpy().ravel()
                    f_var_epistemic_overall_outliers[current_layer] = f_var_epistemic_testing_np[current_layer].numpy().ravel()                        
                    f_var_distributional_overall_outliers[current_layer] = f_var_distributional_testing_np[current_layer].numpy().ravel()

                fig, axs = plt.subplots(nrows = num_layers-1, ncols=2, figsize=(40,20 * (num_layers-1)))
                for current_layer in range(num_layers-1): 

                    current_axs = axs[current_layer,0]


                    current_axs.scatter(Z_np[current_layer+1], np.zeros_like(Z_np[current_layer+1]),
                        s=250, marker="*", alpha=0.95, c = 'cyan',
                        linewidth=1, label = 'Z')
                    
                    current_axs.hist(x=f_mean_overall_inliers[current_layer], bins=50, label = 'pred_mean_inliers', alpha = 0.8)
                    current_axs.hist(x=f_mean_overall_outliers[current_layer], bins=50, label = 'pred_mean_outliers', alpha = 0.8)
                    current_axs.tick_params(axis='both', which='major', labelsize=80)
                    current_axs.legend(loc="upper right",prop={'size': 40})
                    #current_axs.set_xlabel('x') 

                    current_axs = axs[current_layer,1]
                    var_inliers = f_var_epistemic_overall_inliers[current_layer] + f_var_distributional_overall_inliers[current_layer]
                    var_outliers = f_var_epistemic_overall_outliers[current_layer] + f_var_distributional_overall_outliers[current_layer]
                    #current_axs.hist(x=Z_var_np[current_layer+1], bins=50, label = 'Z_var', alpha = 0.8)
                    current_axs.scatter(Z_np[current_layer+1], np.zeros_like(Z_np[current_layer+1]),
                        s=250, marker="*", alpha=0.95, c = 'cyan',
                        linewidth=1, label = 'Z')
                    current_axs.hist(x=var_inliers, bins=50, label = 'pred_var_inliers', alpha = 0.8)
                    current_axs.hist(x=var_outliers, bins=50, label = 'pred_var_outliers', alpha = 0.8)
                    current_axs.tick_params(axis='both', which='major', labelsize=80)
                    current_axs.legend(loc="upper right",prop={'size': 40})

                fig.tight_layout()
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'_moments_detailed.png')
                plt.close()



                #################################################################################
                ###### Get predictive mean and variances at hidden layers for testing set #######
                #################################################################################

                xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
                grid = np.c_[xx.ravel(), yy.ravel()]
                grid = grid.astype(np.float32)

                indices_class_1 = np.where(Y_training==1.0)
                indices_class_0 = np.where(Y_training==0.0)

                print(indices_class_1)
                indices_class_1 = indices_class_1[0]
                indices_class_0 = indices_class_0[0]
                print(indices_class_1)

                points_mean_function = tf.matmul(tf.constant([[-2.0, -1.0],[1.0,1.0]], dtype = DTYPE), W_global)

                correlation_rl_np_overall = defaultdict()
                for plm in range(num_layers):
                    correlation_rl_np_overall[plm] = []
                for nvm in range(100):

                    correlation_rl_np = get_correlation_rl(np.array([[-2.0, -1.0],[1.0,1.0]], dtype = np.float32),  points_mean_function, grid, grid_mean_function, network_architectures_object, propagate_layers_object)
                    for plm in range(num_layers):
                        correlation_rl_np_overall[plm].append(tf.expand_dims(correlation_rl_np[plm], axis = 0))
                for plm in range(num_layers):
                    correlation_rl_np_overall[plm] = tf.concat(correlation_rl_np_overall[plm], axis = 0)
                    correlation_rl_np_overall[plm] = tf.reduce_mean(correlation_rl_np_overall[plm], axis = 0, keepdims = False)
                    correlation_rl_np_overall[plm] = correlation_rl_np_overall[plm].numpy()



                fig, axs = plt.subplots(nrows = 4, ncols=num_layers, sharex = True, sharey = True, figsize=(20 * num_layers, 80 ))

                for current_layer in range(num_layers):

                    current_mean = f_mean_overall[current_layer].numpy()
                    current_mean = inv_probit_np(current_mean)
                    current_mean = current_mean.reshape((100,100))
                    current_var = f_var_overall[current_layer].numpy()
                    current_var = current_var.reshape((100,100))

                    ###################
                    ##### F mean  #####
                    ###################
                    
                    axis = axs[0, current_layer]
                    contour = axis.contourf(xx, yy, current_mean, 50, cmap="coolwarm")
                    cbar1 = fig.colorbar(contour, ax=axis)

                    cbar1.ax.tick_params(labelsize=60) 

                    axis.set(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0),
                            xlabel="$X_1$", ylabel="$X_2$")
                    axis.set_title(label = 'Predictive Mean',fontdict={'fontsize':60})
                    axis.tick_params(axis='both', which='major', labelsize=80)


                    axis.scatter(X_training[indices_class_0,0], X_training[indices_class_0, 1],
                            s=100, marker='X', alpha=0.2, c = 'green',
                            linewidth=1, label='Class 0')
                    axis.scatter(X_training[indices_class_1,0], X_training[indices_class_1, 1],
                            s=100, marker='D', alpha=0.2, c = 'purple',
                            linewidth=1, label = 'Class 1')
                    #axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
                    #        s=750, marker="*", alpha=0.95, c = 'cyan',
                    #        linewidth=1, label = 'Inducing Points')

                    axis.legend(loc="upper right",prop={'size': 60})
                    axis.text(-4.5, 4.5, 'LL:'+"{:.2f}".format(total_nll_np)+'; Acc:'+"{:.2f}".format(precision_testing_overall_np), size=50, color='black')

                    #################################
                    ##### F var Distributional  #####
                    #################################            
                    
                    axis = axs[1, current_layer]
                    contour = axis.contourf(xx, yy, current_var, 50, cmap="coolwarm")
                    cbar1 = fig.colorbar(contour, ax=axis)
                    cbar1.ax.tick_params(labelsize=60) 


                    axis.set(xlim=(-5, 5), ylim=(-5, 5),
                            xlabel="$X_1$", ylabel="$X_2$")

                    axis.set_title(label = 'Distributional Variance',fontdict={'fontsize':60})
                    axis.tick_params(axis='both', which='major', labelsize=80)

                    axis.scatter(X_training[indices_class_0,0], X_training[indices_class_0, 1],
                            s=100, marker='X', alpha=0.2, c = 'green',
                            linewidth=1, label='Class 0')
                    axis.scatter(X_training[indices_class_1,0], X_training[indices_class_1, 1],
                            s=100, marker='D', alpha=0.2, c = 'purple',
                            linewidth=1, label = 'Class 1')

                    #axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
                    #    s=750, marker="*", alpha=0.95, c = 'cyan',
                    #    linewidth=1, label = 'Inducing Points')
                    axis.legend(loc="upper right",prop={'size': 60})




                    ####################################
                    ##### Overcorrelation point 1  #####
                    ####################################
                    axis = axs[2, current_layer]            
                    contours = axis.contour(xx, yy, correlation_rl_np_overall[current_layer][0,:].reshape((100,100)), 10, colors='black')
                    axis.clabel(contours, inline=True, fontsize=32)

                    #plm = axis.imshow(correlation_rl_np_overall[0,:].reshape((100,100)), extent=[-5, 5, -5, 5],
                    #        cmap="coolwarm", alpha=0.5)
                    #cbar1 = fig.colorbar(plm, ax=axis)

                    plm = axis.contourf(xx, yy, correlation_rl_np_overall[current_layer][0,:].reshape((100,100)), 25, cmap="coolwarm", alpha = 0.5)
                    cbar1 = fig.colorbar(plm, ax=axis)

                    cbar1.ax.tick_params(labelsize=60) 


                    axis.set(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0),
                            xlabel="$X_1$", ylabel="$X_2$")
                    axis.set_title(label = 'Correlation',fontdict={'fontsize':60})
                    axis.tick_params(axis='both', which='major', labelsize=80)

                    axis.scatter(-2.0, -1.0,
                            s=500, marker='X', alpha=1.0, c = 'cyan',
                            linewidth=1, label='Focus')

                    axis.scatter(X_training[indices_class_0,0], X_training[indices_class_0, 1],
                            s=100, marker='X', alpha=0.1, c = 'green',
                            linewidth=1, label='Class 0')
                    axis.scatter(X_training[indices_class_1,0], X_training[indices_class_1, 1],
                            s=100, marker='D', alpha=0.1, c = 'purple',
                            linewidth=1, label = 'Class 1')
                    #axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
                    #        s=750, marker="*", alpha=0.95, c = 'cyan',
                    #        linewidth=1, label = 'Inducing Points ')

                    axis.legend(loc="upper right",prop={'size': 60})


                    ####################################
                    ##### Overcorrelation point 2  #####
                    ####################################
                    axis = axs[3, current_layer]            
                    contours = axis.contour(xx, yy, correlation_rl_np_overall[current_layer][1,:].reshape((100,100)), 10, colors='black')
                    axis.clabel(contours, inline=True, fontsize=32)

                    #plm = axis.imshow(correlation_rl_np_overall[0,:].reshape((100,100)), extent=[-5, 5, -5, 5],
                    #        cmap="coolwarm", alpha=0.5)
                    #cbar1 = fig.colorbar(plm, ax=axis)

                    plm = axis.contourf(xx, yy, correlation_rl_np_overall[current_layer][1,:].reshape((100,100)), 25, cmap="coolwarm", alpha = 0.5)
                    cbar1 = fig.colorbar(plm, ax=axis)

                    cbar1.ax.tick_params(labelsize=60) 


                    axis.set(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0),
                            xlabel="$X_1$", ylabel="$X_2$")
                    axis.set_title(label = 'Correlation',fontdict={'fontsize':60})
                    axis.tick_params(axis='both', which='major', labelsize=80)

                    axis.scatter(1.0, 1.0,
                            s=500, marker='X', alpha=1.0, c = 'cyan',
                            linewidth=1, label='Focus')

                    axis.scatter(X_training[indices_class_0,0], X_training[indices_class_0, 1],
                            s=100, marker='X', alpha=0.1, c = 'green',
                            linewidth=1, label='Class 0')
                    axis.scatter(X_training[indices_class_1,0], X_training[indices_class_1, 1],
                            s=100, marker='D', alpha=0.1, c = 'purple',
                            linewidth=1, label = 'Class 1')
                    #axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
                    #        s=750, marker="*", alpha=0.95, c = 'cyan',
                    #        linewidth=1, label = 'Inducing Points ')

                axis.legend(loc="upper right",prop={'size': 60})


                plt.tight_layout()
                plt.savefig('./figures/'+where_to_save+'/plot_iteration_'+str(i)+'_orizontal.png')
                plt.close()



            ###############################################################################
            ####################### End ###################################################
            ###############################################################################


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inducing', type=int, help = 'the number of inducing points at each GP in the DGP')
    args = parser.parse_args()	


    df = pd.read_csv('./datasets/banana.csv', header=None)
    data = df.values
    X_data = data[:,:2]
    Y_data = data[:,-1].reshape((-1,1)) - 1.0
    
    plt.scatter(x =X_data[:,0], y = X_data[:,1], c = Y_data.ravel())
    plt.show()

   # plt.scatter(x,y)
    plt.savefig('./banana_dataset.png')
    plt.close()

    np.random.seed(7)
    lista = np.arange(X_data.shape[0])
    np.random.shuffle(lista)
    index_training = lista[:4000]
    index_testing = lista[4000:]

    x_values_training_np = X_data[index_training,...]
    y_values_training_np = Y_data[index_training,...]
    
    print('size of training dataset')
    print(x_values_training_np.shape)
    print(y_values_training_np.shape)
    x_values_testing_np = X_data[index_testing,...]
    y_values_testing_np = Y_data[index_testing,...]
    
    print('size of testing dataset')
    print(x_values_testing_np.shape)
    print(y_values_testing_np.shape)

    x_values_training_np = x_values_training_np.reshape((-1,2))
    x_values_testing_np = x_values_testing_np.reshape((-1,2))

    y_values_training_np = y_values_training_np.reshape((-1,1))
    y_values_testing_np = y_values_testing_np.reshape((-1,1))


    x_values_training_np = x_values_training_np.astype(np.float32)
    x_values_testing_np = x_values_testing_np.astype(np.float32)

    y_values_training_np = y_values_training_np.astype(np.float32)
    y_values_testing_np = y_values_testing_np.astype(np.float32)

    num_inducing = args.num_inducing

    km = KMeans(n_clusters=num_inducing).fit(x_values_training_np)
    k_mean_output = km.cluster_centers_

    num_layers = 3
    dim_layers = [2, 1, 1, 1]
    num_inducing = [num_inducing for _ in range(num_layers)]

    main_DeepGP(num_data = x_values_training_np.shape[0], dim_input = x_values_training_np.shape[1], dim_output = 1,
        num_iterations = 10001, num_inducing = num_inducing, 
        type_var = 'full', num_layers = num_layers, dim_layers = dim_layers,
        num_batch=32, Z_init = k_mean_output,  
        num_test = x_values_testing_np.shape[0], learning_rate = 1e-3, 
        base_seed = 0, mean_Y_training = None, dataset_name = 'ddgp_banana',
        use_diagnostics = True, task_type = 'classification', 
        X_training = x_values_training_np, 
        Y_training = y_values_training_np, 
        X_testing = x_values_testing_np, 
        Y_testing = y_values_testing_np)



