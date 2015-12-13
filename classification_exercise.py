from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)


X_digits_train = X_digits[:0.9*n_samples]
y_digits_train = y_digits[:0.9*n_samples]

X_digits_test = X_digits[0.9*n_samples:]
y_digits_test = y_digits[0.9*n_samples:]

regr = linear_model.LogisticRegression(C=1e5)
print('Logistic Regression score: ',regr.fit(X_digits_train, y_digits_train).score(X_digits_test,y_digits_test))

knn = neighbors.KNeighborsClassifier()
print('K Nearest Neighbors score: ',knn.fit(X_digits_train, y_digits_train).score(X_digits_test,y_digits_test))




