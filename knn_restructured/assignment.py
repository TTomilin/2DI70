# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:51:07 2020

@author: Tristan
"""

from sklearn.metrics import accuracy_score

from knn import KNN
from utils import *

train = read_file('data/MNIST_train_small.csv')
test = read_file('data/MNIST_test_small.csv')

"""
Excercise 1a) - Write down your implementation of k-NN neighbors 
(using as training data MNIST train small.csv) and report on its accuracy to 
predict the labels in both the training and test sets (respectively MNIST train
 small.csv and MNIST test small.csv). For this question use the simple Euclidean distance. 
Make a table of results for k ∈ {1,...,20}, plot your the empirical training 
and test loss as a function of k, and comment on your results.
 Explain how ties are broken in Equation 1.
"""

a_knn_train_train = KNN(train, minkowski_dist_f(2))
a_knn_train_train.initalize_nn_list(train)

a_knn_train_test = KNN(test, minkowski_dist_f(2))
a_knn_train_test.initalize_nn_list(test)

# training_acurracies = []
k_list = [k for k in range(1, 21)]
test_accuracies = {key: None for key in k_list}

# training_labels = list(train["label"].values)
test_labels = list(test["label"].values)

for k in k_list:
    # training_predictions = a_knn_train_train.predict(list(train.index), k=k, loocv=False)
    # train_acc = accuracy_score(training_labels, training_predictions, normalize=True, sample_weight=None)
    # training_acurracies.append(train_acc)

    test_predictions = a_knn_train_test.predict(list(test.index), k=k, loocv=False)
    test_acc = accuracy_score(test_labels, test_predictions, normalize=True, sample_weight=None)
    test_accuracies[k - 1] = test_acc

# MAKE ME BEAUTIFUL!
# plot_accuracy(training_acurracies, "Accuracy", k_list)
# plot_accuracy(test_accuracies, "Accuracy", k_list)

"""
Excercise 1b) - Obviously the choice of the number of neighbors k is crucial 
    to obtain good performance. This choice must be made WITHOUT LOOKING at the
    test dataset. Although one can use rules-of-thumb, a possibility is to use
    cross-validation. LeaveOne-Out Cross-Validation (LOOCV) is extremely simple 
    in our context. Implement LOOCV to estimate the risk of the k-NN rule 
    for k ∈ {1,...,20}. Report these LOOCV risk estimates4 on a table and plot
    them as well the empirical loss on the test dataset (that you obtained in (a)). 
    Given your results, what would be a good choice for k? Comment on your results.

"""

"""
loocv_risk_estimates = []

for k in k_list:       
    training_predictions = a_knn_train_train.predict(list(train.index), k=k, loocv=True)
    training_labels = list(train["label"].values) 
    #implement risk_estimate
    #loocv_risk_estimate = risk_estimate(training_predictions, training_labels)
    #plot it here
"""

"""
Excercise 1c) -  Obviously, the choice of distance metric also plays an 
    important role. Consider a simple generalization of the Euclidean distance,
    namely `p distances (also known as Minkowski distances). For x,y ∈Rl deﬁne
    dp(x,y) = l X i=1 |xi −yi|p!1/p , where p ≥ 1. Use leave-one-out cross validation
    to simultaneously choose a good value for k ∈{1,...,20} and p ∈ [1,15].

"""

"""
p_list = [p for p in range(1,3)]

accuracy_dict = {}

#training_labels:
for p in p_list:
    accuracy_dict["p_"+str(p)] = {}
    c_knn = KNN()
    c_knn.load("models/knn_small_train_train_minkowski_p_"+str(p)+".pkl")
    for k in k_list:
        c_knn_predictions = c_knn.predict(list(train.index), k=k, loocv=True)
        accuracy = accuracy_score(training_labels, c_knn_predictions, normalize=True, sample_weight=None)
        accuracy_dict["p_"+str(p)]["k_"+str(k)] = accuracy
"""
