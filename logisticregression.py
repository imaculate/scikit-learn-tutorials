from sklearn.linear_model import LogisticRegressionCV
import numpy as np

n_samples, n_features = 50, 5
rng = np.random.RandomState(0)
X_ref = rng.randn(n_samples, n_features)
y = rng.choice(['foo', 'bar', 'baz'], n_samples)
X_ref -= X_ref.mean()
X_ref /= X_ref.std()
lr_cv = LogisticRegressionCV(Cs=[1.], fit_intercept=False, multi_class='multinomial')
lr_cv.fit(X_ref, y)