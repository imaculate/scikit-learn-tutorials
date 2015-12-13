from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import pickle

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
joblib.dump(clf, 'filename.pkl')
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
