import scipy as sp
from scipy import misc
import numpy as np
from sklearn import cluster
try:
    lena = sp.lena()
except AttributeError:
    lena = misc.lena()
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

print(values)

lena_compressed = np.choose(labels, values)

lena_compressed.shape = lena.shape
