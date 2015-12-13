# Create a signal with only 2 useful dimensions
import numpy as np
from sklearn import decomposition
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]
print(X.shape)

pca = decomposition.PCA()
pca.fit(X)
print(pca.explained_variance_)


# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)