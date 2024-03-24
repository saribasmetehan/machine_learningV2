import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate( (x1, x2, x3), axis = 0)
y = np.concatenate( (y1, y2, y3), axis = 0)


dictionary = {"x": x,"y": y}

data = pd.DataFrame(dictionary)


merg = linkage(data,method="ward")

dendrogram(merg,leaf_rotation=90)

plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

hiyerartical_cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

cluster = hiyerartical_cluster.fit_predict(data)

data["label"] = cluster

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="yellow")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="blue")

plt.show()

































