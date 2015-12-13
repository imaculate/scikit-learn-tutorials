from sklearn.grid_search import GridSearchCV
from sklearn import datasets,svm,cross_validation
import numpy as np
def code():
    Cs = np.logspace(-6, -1, 10)
    svc = svm.SVC(kernel='linear')
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)

    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    clf.fit(X_digits[:1000], y_digits[:1000])

    print(clf.best_estimator_.C)


    print(clf.score(X_digits[1000:], y_digits[1000:]))
    print(cross_validation.cross_val_score(clf, X_digits, y_digits))
    

if __name__ == '__main__':
    code()
