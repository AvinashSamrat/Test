# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:46:04 2020

@author: avinash
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Data Sets
FileName="IELTS.csv"
data=pd.read_csv(FileName)

X=data.iloc[:,5:7].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means Algo to Dataset
kmeans=KMeans(n_clusters=4,random_state=40)
y_kmeans=kmeans.fit_predict(X)


# Visualising the Clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label="Cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title("Cluster of Students")
plt.xlabel("Bands")
plt.ylabel("Overall Score")
plt.legend()
plt.show()

