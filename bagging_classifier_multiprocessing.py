from sklearn.ensemble import BaggingClassifier
from sklearn import datasets


if __name__ == '__main__':
    data = datasets.load_digits()
    X_train = data.data[:-20]
    y_train = data.target[:-20]
    X_test = data.data[-20:]
    y_test = data.target[-20:]
    for num in range(1,6):
        clf = BaggingClassifier(n_estimators=num, n_jobs=4)
        clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        print(num,score)