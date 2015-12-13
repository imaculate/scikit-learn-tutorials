from __future__ import print_function
from sklearn import datasets
import numpy as np
from sklearn import linear_model

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.Ridge()
alphas = np.logspace(-4, -1, 6)
print(regr.get_params().keys())
print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])

regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test)for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)