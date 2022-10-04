# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:55:18 2020

@author: Tristan
"""

from knn_restructured.knn import KNN
from knn_restructured.utils import *

train = read_file('data/MNIST_train_small.csv')
test = read_file('data/MNIST_test_small.csv')
p_list = [p for p in range(1, 2)]

if __name__ == '__main__':

    for p in p_list:
        knn = KNN(train, minkowski_dist_f(p))
        print("Computing KNN, for mink. dist, p=" + str(p))
        knn.initalize_nn_list(test)
        knn.save("models/test_knn_small_train_train_minkowski_p_" + str(p) + ".pkl")
