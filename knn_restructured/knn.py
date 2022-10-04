# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:46:10 2020

@author: Tristan
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


class KNN:
    training_data = []
    test_data = []
    distance_f = []

    def __init__(self, data=None, distance_f=None, max_k=30):
        self.training_data = data
        self.distance_f = distance_f
        self.max_k = max_k

    def initalize_nn_list(self, data):
        self.test_data = data
        self.distance_dict = pd.DataFrame()

        image_list_matrix = np.asarray(self.training_data.iloc[:, 1:]).T

        for image_inx, image_row in tqdm(self.test_data.iterrows(), total=self.test_data.shape[0]):
            image_image = image_row[1:]
            image_single_matrix = np.asarray(image_image).reshape(self.test_data.shape[1] - 1, 1)
            dist = self.distance_f(image_list_matrix, image_single_matrix)
            dist_label = np.c_[dist, self.training_data['label']]  # append the training labels: [distance, label]
            dist_label_sort = dist_label[dist_label[:, 0].argsort(kind='quicksort')]  # sort the distances
            first_k_labels = dist_label_sort[:self.max_k, 1]
            self.distance_dict[int(image_inx)] = first_k_labels

    def predict(self, index_list, k=20, loocv=False):
        # if loocv = True, it leaves out the closest neighbour
        slicer = slice(1, k + 1) if loocv else slice(k)
        return [int(stats.mode(self.distance_dict[index].values[slicer])[0][0]) for index in index_list]

    def save(self, file_name):
        pickle.dump(self.distance_dict, open(file_name, "wb"))

    def load(self, file_name):
        self.distance_dict = pickle.load(open(file_name, "rb"))
