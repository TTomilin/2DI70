# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:50:36 2020

@author: Tristan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
Why does sklearn have a Knn fit() function?
- https://stats.stackexchange.com/questions/349842/why-do-we-need-to-fit-a-k-nearest-neighbors-classifier

Algorithm for K-NN:
1.   Initialize k
2.   For each row:
    1. Calculate distance between the data point and each data points in the given data file.
    2. Then add the distances corresponding to rows in given data file (probably by adding column for distance).
    3. Sort the data in data file from smallest to largest (in ascending order) by the distances.
3.   Pick the first K entries from the sorted collection of data.
4.   Observe the labels of the selected K entries.
5.   For classification, return the mode of the K labels and for regression, return the mean of K labels.
'''


def read_file(file_name):
    return pd.read_csv(file_name, header=None, names=['label', *np.arange(start=1, stop=785, step=1)])


def minkowski_dist(v, w, p=2):
    D = np.abs(np.subtract(w, v))
    Dp = np.power(D, p)
    distance_vals = np.sum(Dp, axis=0)
    distance_vals_p = np.power(distance_vals, 1 / p)
    return distance_vals_p


def minkowski_dist_f(p=2):
    def minkowski_dist(v, w, p=2):
        D = np.abs(np.subtract(w, v))
        Dp = np.power(D, p)
        distance_vals = np.sum(Dp, axis=0)
        distance_vals_p = np.power(distance_vals, 1 / p)
        return distance_vals_p

    return minkowski_dist


def minkowski_distances(v, w, p_list=[1, 2, 3]):
    D = np.abs(np.subtract(w, v))
    dist_list = []
    for p in p_list:
        Dp = np.power(D, p)
        distance_vals = np.sum(Dp, axis=0)
        distance_vals_p = np.power(distance_vals, 1 / p)
        dist_list.append(distance_vals_p)
    return dist_list


def cosine(v, w):
    D = np.dot(w.T, v)
    return 2 - 2 * D.T


def plot_empirical_risk(emp_risks, param_name, param_range, scale='linear', ylim=(0, 1), ylabel='risk'):
    """ Renders a plot that updates with every evaluation from evaluator.
    Keyword arguments:
    emp_risks -- list of empirical risk values
    param_name -- the parameter that is being varied on the X axis. Can be a hyperparameter, sample size,...
    param_range -- list of all possible values on the x-axis
    scale -- defines which scale to plot the x-axis on, either 'log' (logarithmic) or 'linear'
    ylim -- tuple with the lowest and highest y-value to plot (e.g. (0, 10))
    ylabel -- the y-axis title
    """
    # Plot interactively
    plt.ion()
    plt.ylabel(ylabel)
    plt.xlabel(param_name)

    # Make the scale look nice
    plt.xscale(scale)
    plt.xlim(param_range[0], param_range[-1])
    plt.ylim(ylim)

    # Start from empty plot, then fill it
    series = {}
    lines = {}
    xvals = []
    for i in param_range:
        scores = emp_risks
        if i == param_range[0]:  # initialize series
            for k in scores.keys():
                # lines[k], = plt.plot([i[0] for i in scores], [], marker = '.', label = k)
                lines[k], = plt.plot(xvals, [], marker='.', label=k)
                series[k] = []
        xvals.append(i)
        for k in scores.keys():  # append new data
            series[k].append(scores[k][i - 1])
            lines[k].set_data(xvals, series[k])
        # refresh plot
        plt.legend(loc='best')
        plt.margins(0.1)
        display.display(plt.gcf())
        display.clear_output(wait=True)


def get_emp_risk(labels, predictions):
    n = len(labels)  # number of images
    losses = np.not_equal(labels, predictions)

    sum_loss = losses.sum(axis=0)  # sum up losses for all images for one k value
    emp_risk = sum_loss / n  # the average loss

    return emp_risk


def get_pca_data(train, test):
    ss = StandardScaler()
    standardized_train = ss.fit_transform(train.iloc[:, 1:])
    standardized_test = ss.transform(test.iloc[:, 1:])

    pca = PCA(n_components=764, svd_solver='arpack')
    transformed_train = pca.fit_transform(standardized_train)
    transformed_test = pca.transform(standardized_test)

    df_train = pd.DataFrame(data=transformed_train, columns=np.arange(1, 765, 1))
    df_train.insert(loc=0, column='label', value=train['label'])
    df_test = pd.DataFrame(data=transformed_test, columns=np.arange(1, 765, 1))
    df_test.insert(loc=0, column='label', value=test['label'])

    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)

    return df_train, df_test, cum_var_explained
