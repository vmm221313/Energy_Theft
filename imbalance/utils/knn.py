import numpy as np 
import operator
import utils
import multiprocessing as mp
from tqdm import tqdm

class GetNeighbors:
    def dist_pool_funk(self, i):

        # calculate the distance between the test instance and each training instance
        if self.pre_computed_matrix is None: 
            dist , _ = self.dist_fun(self.x_test_instance, self.x_train[i],**self.dist_fun_params)
        else: 
            # do not re-compute the distance just get it from the precomputed one
            dist = self.pre_computed_matrix[i, self.index_test_instance]
        # add the index of the current training instance and its corresponding distance 
        return (i, dist)

    def get_neighbors(self, x_train, x_test_instance, k, dist_fun, dist_fun_params, 
                    pre_computed_matrix=None, index_test_instance=None,
                    return_distances = False): 
        """
        Given a test instance, this function returns its neighbors present in x_train
        NB: If k==0 zero it only returns the distances
        """


        self.x_train                = x_train
        self.x_test_instance        = x_test_instance
        self.dist_fun               = dist_fun
        self.dist_fun_params        = dist_fun_params
        self.index_test_instance    = index_test_instance
        self.pre_computed_matrix    = pre_computed_matrix

        pool = mp.Pool()
        print('Calculating Distances')
        distances = list(tqdm(p.imap(dist_pool_funk, range(len(x_train)))))

        '''
        # loop through the training set 
        for i in range(len(x_train)): 
            # calculate the distance between the test instance and each training instance
            if pre_computed_matrix is None: 
                dist , _ = dist_fun(x_test_instance, x_train[i],**dist_fun_params)
            else: 
                # do not re-compute the distance just get it from the precomputed one
                dist = pre_computed_matrix[i,index_test_instance]
            # add the index of the current training instance and its corresponding distance 
            distances.append((i, dist))
        '''
        
        # if k (nb_neighbors) is zero return all the items with their distances 
        # NOT SORTED 
        if k==0: 
            if return_distances == True: 
                return distances
            else:
                print('Not implemented yet')
                exit()
        # sort list by specifying the second item to be sorted on 
        distances.sort(key=operator.itemgetter(1))
        # else do return only the k nearest neighbors
        neighbors = []
        for i in range(k): 
            if return_distances == True: 
                # add the index and the distance of the k nearest instances from the train set 
                neighbors.append(distances[i])
            else:
                # add only the index of the k nearest instances from the train set 
                neighbors.append(distances[i][0])
            
        return neighbors
   
    