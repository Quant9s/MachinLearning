from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np

cancer = load_breast_cancer()
data = np.array(cancer.data)
target = np.array(cancer.target)

cndt_train = data[:500]
rslt_train = target[:500]
cndt_test = data[500:]
rslt_test = target[500:]

knn = KNN(n_neighbors=5)
knn.fit(cndt_train,rslt_train)
rslt_pred = knn.predict(cndt_test)

accuracy = np.sum(np.equal(rslt_pred,rslt_test))/len(rslt_test)

print(accuracy)