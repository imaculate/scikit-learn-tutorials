from sklearn import linear_model
from sklearn import datasets
import numpy as np
from sklearn.utils.testing import *
from numpy.testing import *

ransac = linear_model.RANSACRegressor()
digits = datasets.load_diabetes()
X = digits.data
y = digits.target
n_samples = y.shape[0]


weights = np.ones(n_samples)
ransac.fit(X, y, weights)

print(ransac.inlier_mask_.shape)
assert_equal(ransac.inlier_mask_.shape[0], n_samples)


ransac2 = linear_model.RANSACRegressor(linear_model.Lasso())

assert_raises(TypeError,ransac2.fit,X,y, weights )


