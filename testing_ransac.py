from sklearn import linear_model
from sklearn import datasets
import numpy as np

ransac = linear_model.RANSACRegressor()
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
ransac.fit(X_digits, y_digits, sample_weight=np.arange(y_digits.shape[0]))
print(ransac._estimator_type)