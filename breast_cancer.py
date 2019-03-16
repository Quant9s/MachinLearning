from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
data = np.array(cancer.data)
target = np.array(cancer.target)
cndt_train = data[:500]
rslt_train = target[:500]
cndt_test = data[500:]
rslt_test = target[500:]

### 비교군의 상태와 시험군의 상태 거리

def get_eudists(trains,tests):
    dists = []
    for _test in tests:
        dist = []
        for _train in trains:
            dist.append(np.linalg.norm(_train-_test))
        dists.append(dist)
    return np.array(dists)

### 시험군과 가장 가까운 비교군의 결과로 결과 예측 데이터들 

def get_prediction(_eudists,_train,k):
    sort_eudists = _eudists.argsort()
    _preds_list = _train[sort_eudists][:,:k]
    _pred = [np.bincount(_preds).argmax() 
           for _preds in _preds_list]
    return _pred

### 예측 값의 정확도 측정

def get_accuracy(_pred,_test):
    return np.sum(np.equal(_pred,_test))/len(_test)

#   

cndt_eudists = np.array(get_eudists(cndt_train,cndt_test))

k_val = 5
rslt_pred = get_prediction(cndt_eudists,rslt_train,k_val)

accuracy = get_accuracy(rslt_pred,rslt_test)
print(accuracy)