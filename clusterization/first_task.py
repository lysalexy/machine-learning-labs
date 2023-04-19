import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def draw_graph(x, y, xlabel, ylabel, title, is_scaley_needed):
    plt.plot(x, y,scaley=is_scaley_needed)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def test_max_iter_kmeans_dependency(data,max_iter):
    kmeans = KMeans(init="k-means++", n_clusters=3, max_iter = max_iter )
    kmeans.fit(data)
    return kmeans

def check_davies_bouldin_score(kmeans,data):
    labels = kmeans.labels_
    return davies_bouldin_score(data,labels)

def draw_clustarization_results(data,cluster_method):
    pca = PCA(2)
    df = pca.fit_transform(data)
    labels = cluster_method.labels_
    u_labels = np.unique(labels)

    for i in u_labels:
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
    plt.legend()
    plt.show()

def test_data_clasterization(data):
    max_iter = []
    davies_bouldin_scores = []
    for i in range(10, 1000, 10):
        max_iter.append(i)
        kmeans = test_max_iter_kmeans_dependency(data, i)
        davies_bouldin_scores.append(check_davies_bouldin_score(kmeans, data))
    kmeans = test_max_iter_kmeans_dependency(data, 300)
    draw_graph(max_iter, davies_bouldin_scores, 'max iter', 'David-Bouldin score', '', True)
    draw_clustarization_results(data, kmeans)

def get_standaridized_data(data):
    cols_to_standardize = [
        column for column in data.columns
    ]
    data_to_standardize = data[cols_to_standardize]
    scaler = StandardScaler().fit(data_to_standardize)

    standardized_data = data.copy()
    standardized_columns = scaler.transform(data_to_standardize)
    standardized_data[cols_to_standardize] = standardized_columns
    return standardized_data


def first_task():
    data = pd.read_csv(r'files\pluton.csv ')
    test_data_clasterization(data)
    stand_data=get_standaridized_data(data)
    test_data_clasterization(stand_data)
