import numpy as np
from sklearn import datasets,svm,cross_validation
def code():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    """X_folds = np.array_split(X_digits, 3)
    y_folds = np.array_split(y_digits, 3)"""
    svc = svm.SVC(C=1, kernel='linear')
    """scores = list()


    for k in range(3):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test  = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

    print(scores)"""

    k_fold = cross_validation.KFold(len(X_digits), n_folds=3)
    """for train_indices, test_indices in k_fold:
        print('Train: %s | test: %s' % (train_indices, test_indices))"""


    #print([svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold])

    print(cross_validation.cross_val_score(svc, X_digits, y_digits, cv=k_fold,n_jobs=-1))

if __name__ == '__main__':
    code()
