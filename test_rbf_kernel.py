import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import fetch_mldata
import sys

X = []
y = []
for i in np.linspace(0, 10, 10):
    X.append([i])
    y.append(i * 2)

# display data
a = input('Do you want to display test data ? ')
if a == 'y':
    print('X: ', X)
    print('y: ', y)

# initialize the kernel
kernel_test = RBF(length_scale = [2])

print('after kernel init')

# train the model
gp_test = GaussianProcessRegressor(kernel=kernel_test, alpha=0, normalize_y=True)

print('after GP regressor')

gp_test.fit(X, y)

print("GPML kernel: %s" % gp_test.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp_test.log_marginal_likelihood(gp_test.kernel_.theta))


X_ = []
for i in range(15):
    X_.append([i + 0.5])
y_pred, y_std = gp_test.predict(X_, return_std=True)

# Plot the predict result
X = np.array(X)
y = np.array(y)
X_ = np.array(X_)
plt.scatter(X[:, 0], y, c='k')
plt.plot(X_[:, 0], y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std, alpha = 0.5, color='k')

plt.xlim(X_[:, 0].min(), X_[:, 0].max())
plt.xlabel("x")
plt.ylabel(r"u")
plt.title(r"Test")
plt.tight_layout()
plt.show()