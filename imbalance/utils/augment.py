# this contains the data generation methods of icdm 2017 
# "Generating synthetic time series to augment sparse datasets"
import numpy as np
import random
from tqdm import tqdm
import pickle

from utils.distances.dtw import dynamic_time_warping as dtw
#from utils.dba import calculate_dist_matrix
from utils.dba import DistanceMatrix
from utils.dba import dba 
from utils.knn import GetNeighbors

GN = GetNeighbors()

import multiprocessing as mp

# weights calculation method : Average Selected (AS)
def get_weights_average_selected(x_train, dist_pair_mat, distance_algorithm='dtw'):
    # get the distance function 
    dist_fun = DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    # get the number of dimenions 
    num_dim = x_train[0].shape[1]
    # number of time series 
    n = len(x_train)
    # maximum number of K for KNN 
    max_k = 5 
    # maximum number of sub neighbors 
    max_subk = 2
    # get the real k for knn 
    k = min(max_k,n-1)
    # make sure 
    subk = min(max_subk,k)
    # the weight for the center 
    weight_center = 0.5 
    # the total weight of the neighbors
    weight_neighbors = 0.3
    # total weight of the non neighbors 
    weight_remaining = 1.0- weight_center - weight_neighbors
    # number of non neighbors 
    n_others = n - 1 - subk
    # get the weight for each non neighbor 
    if n_others == 0 : 
        fill_value = 0.0
    else:
        fill_value = weight_remaining/n_others
    # choose a random time series 
    idx_center = random.randint(0,n-1)
    # get the init dba 
    init_dba = x_train[idx_center]
    # init the weight matrix or vector for univariate time series 
    weights = np.full((n,num_dim),fill_value,dtype=np.float64)
    # fill the weight of the center 
    weights[idx_center] = weight_center
    # find the top k nearest neighbors
    topk_idx = np.array(GN.get_neighbors(x_train,init_dba,k,dist_fun,dist_fun_params, pre_computed_matrix = dist_pair_mat, index_test_instance = idx_center))
    # select a subset of the k nearest neighbors 
    final_neighbors_idx = np.random.permutation(k)[:subk]
    # adjust the weight of the selected neighbors 
    weights[topk_idx[final_neighbors_idx]] = weight_neighbors / subk
    # return the weights and the instance with maximum weight (to be used as 
    # init for DBA )
    return weights, init_dba

class DataAugmentation:
    def one_aug_step(self, n):
        # get the weights and the init for avg method 
        weights, init_avg = self.weights_fun(self.c_x_train, self.dist_pair_mat, distance_algorithm=self.distance_algorithm)
        # get the synthetic data 
        synthetic_mts = dba(self.c_x_train, dba_iters, verbose=False, 
                        distance_algorithm=self.distance_algorithm,
                        weights=weights,
                        init_avg_method = 'manual',
                        init_avg_series = init_avg)  
        
        return synthetic_mts

    def augment_train_set(self, x_train, y_train, classes, N, dba_iters=5, limit_N = True):
        """
        This method takes a dataset and augments it using the method in icdm2017. 
        :param x_train: The original train set
        :param y_train: The original labels set 
        :param N: The number of synthetic time series. 
        :param dba_iters: The number of dba iterations to converge.
        :param weights_method_name: The method for assigning weights (see constants.py)
        :param distance_algorithm: The name of the distance algorithm used (see constants.py)
        """
        self.distance_algorithm = 'dtw'

        # get the weights function
        self.weights_fun = get_weights_average_selected
        # get the distance function 
        dist_fun = dtw
        # get the distance function params 
        dist_fun_params = {'w':-1} # warping window should be given in percentage (negative means no warping window)

        # synthetic train set and labels 
        synthetic_x_train = []
        synthetic_y_train = []
        
        # loop through each class
        for c in classes: 
            print('Generating samples for class {}'.format(c))

            # get the MTS for this class 
            self.c_x_train = x_train[np.where(y_train==c)]

            if len(self.c_x_train) == 1 :
                # skip if there is only one time series per set
                continue

            if limit_N == True:
                # limit the nb_prototypes
                nb_prototypes_per_class = min(N, len(self.c_x_train))
            else:
                # number of added prototypes will re-balance classes
                nb_prototypes_per_class = N + (N-len(self.c_x_train))

            # get the pairwise matrix 
            #print('Calculating Distance Matrix')
            #d_mat_obj       = DistanceMatrix()
            #dist_pair_mat   = d_mat_obj.calculate_dist_matrix(c_x_train,dist_fun,dist_fun_params)
            self.dist_pair_mat    = None

            #with open('data/processed/dist_pair_mat.pkl', 'wb') as f:
            #    pickle.dump(dist_pair_mat, f)

            print('Starting DBA Loop')
            print(nb_prototypes_per_class)
            
            p = mp.Pool()

            synthetic_x_train = list(tqdm(map(self.one_aug_step, range(nb_prototypes_per_class)), total=nb_prototypes_per_class))
            synthetic_y_train += [c]*nb_prototypes_per_class
            
            '''
            # loop through the number of synthtectic examples needed
            for n in tqdm(range(nb_prototypes_per_class)): 
                # get the weights and the init for avg method 
                weights, init_avg = weights_fun(c_x_train,dist_pair_mat,
                                                distance_algorithm='dtw')
                # get the synthetic data 
                synthetic_mts = dba(c_x_train, dba_iters, verbose=False, 
                                distance_algorithm='dtw',
                                weights=weights,
                                init_avg_method = 'manual',
                                init_avg_series = init_avg)  
                # add the synthetic data to the synthetic train set
                synthetic_x_train.append(synthetic_mts)
                # add the corresponding label 
                synthetic_y_train.append(c)
            '''

        # return the synthetic set 
        return np.array(synthetic_x_train), np.array(synthetic_y_train)       

AVERAGING_ALGORITHMS = {'dba':dba}

DISTANCE_ALGORITHMS = {'dtw': dtw}

DTW_PARAMS = {'w':-1} # warping window should be given in percentage (negative means no warping window)

DISTANCE_ALGORITHMS_PARAMS = {'dtw':DTW_PARAMS}

MAX_PROTOTYPES_PER_CLASS = 5

WEIGHTS_METHODS = {'as':get_weights_average_selected }
        
    
    

