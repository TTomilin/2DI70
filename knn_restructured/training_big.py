# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:30:14 2020

@author: Tristan
"""

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from knn_restructured.knn import KNN
from knn_restructured.utils import *

train = read_file('data/MNIST_train.csv')
test = read_file('data/MNIST_test.csv')


def get_pca_data(train, test):
    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]

    standardized_train = StandardScaler().fit_transform(train)
    standardized_test = StandardScaler().fit_transform(test)

    pca = PCA(n_components=764, svd_solver='arpack')
    transformed_train = pca.fit_transform(standardized_train)
    transformed_test = pca.transform(standardized_test)

    df_train = pd.DataFrame(data=transformed_train, columns=np.arange(1, 765, 1))
    df_train.insert(loc=0, column='label', value=train['label'])
    df_test = pd.DataFrame(data=transformed_test, columns=np.arange(1, 765, 1))
    df_test.insert(loc=0, column='label', value=test['label'])

    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
    cum_var_explained = np.cumsum(percentage_var_explained)

    return df_train, df_test, cum_var_explained


def normalize(dataset):
    for image_inx, image_row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        image_row[1:] = np.float32(image_row[1:] / np.linalg.norm(image_row[1:]))
        dataset.iloc[image_inx] = image_row
    return None


print("normalize train data")
normalize(train)
print("normalize test data")
normalize(test)
# train = get_pca_data(train)
# test = get_pca_data(test)


if __name__ == '__main__':

    knn = KNN(train, cosine)
    print("fit KNN")
    knn.initalize_nn_list(train)
    knn.save("models/knn-big_train_euclidean.pkl")
    knn.load("models/knn-big_train_euclidean.pkl")

    for k in range(1, 10):
        predictions = knn.predict(list(train.index), k=k, loocv=True)
        labels = list(train["label"].values)
        acc = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
        print(acc)
