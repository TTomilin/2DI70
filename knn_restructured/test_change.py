# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:55:18 2020

@author: Tristan
"""

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from knn_restructured.knn import KNN
from knn_restructured.utils import *

train = read_file('data/MNIST_train_small.csv')
test = read_file('data/MNIST_test_small.csv')

knn_test = KNN(train, minkowski_dist_f(p=2))
knn_test.initalize_nn_list(test)


def normalize(dataset):
    for image_inx, image_row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        image_row[1:] = np.float32(image_row[1:] / np.linalg.norm(image_row[1:]))
        dataset.iloc[image_inx] = image_row
    return None


print("normalize train data")
normalize(train)
print("normalize test data")
normalize(test)

knn = KNN(train, cosine)
knn.initalize_nn_list(test)

for k in range(1, 20):
    predictions = knn.predict(list(test.index), k)
    labels = list(test["label"].values)
    acc = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
    print(acc)
