from sklearn import datasets, linear_model

iris = datasets.load_iris()
print(iris.DESCR)
iris_X_train = iris.data[:-20]
iris_y_train = iris.target[:-20]
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)
