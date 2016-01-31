import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
data_train = np.loadtxt('data.csv', delimiter=',')
X = data_train[:, 0:2]
y = data_train[:, 3]
clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
data_test = np.loadtxt('dataimg.csv', delimiter=',')
img = data_test[:, 0:2]
# make predictions
expected = y
predicted = clf.predict(img)
print predicted
# summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
