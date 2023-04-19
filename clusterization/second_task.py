from first_task import draw_clustarization_results

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

def test_kmeans(data,clusters_amount):
    kmeans = KMeans(init="k-means++", n_clusters=clusters_amount)
    kmeans.fit(data)
    return kmeans

def test_dbscan(data):
    dbscan = DBSCAN()
    dbscan.fit(data)
    return dbscan

def test_agglomerative_clastering(data,clusters_amount):
    aggl = AgglomerativeClustering(n_clusters=clusters_amount)
    aggl.fit(data)
    return aggl

def test_clastarizatiion(data, clusters_amount):
    kmean = test_kmeans(data, clusters_amount)
    draw_clustarization_results(data, kmean)
    agg = test_agglomerative_clastering(data, clusters_amount)
    draw_clustarization_results(data, agg)
    dbscan = test_dbscan(data)
    labels = dbscan.labels_
    u_labels = np.unique(labels)
    clusters_amount = len(u_labels)
    print(clusters_amount)
    draw_clustarization_results(data, dbscan)

def second_task():
    data_1 = pd.read_csv(r'files\clustering_1.csv', sep='\t')
    test_clastarizatiion(data_1,2)

    data_2 = pd.read_csv(r'files\clustering_2.csv',sep='\t')
    test_clastarizatiion(data_2,3)

    data_3 = pd.read_csv(r'files\clustering_3.csv',sep='\t')
    test_clastarizatiion(data_3,3)