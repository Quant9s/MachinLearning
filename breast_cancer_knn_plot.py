from sklearn.datasets import load_breast_cancer 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import sys, traceback

import seaborn as sns 
import matplotlib.pyplot as plt

class _knnOneResult:
    def __init__(self,disease_data):
        self.cancer = disease_data
        self.data   = np.array(self.cancer.data)
        self.target = np.array(self.cancer.target)
    
    def dataSetRatio(self,ratio):
        len_data = int(len(self.data)*ratio)
        
        cndt_train = self.data[:len_data]
        rslt_train = self.target[:len_data]
        cndt_test  = self.data[len_data:]
        rslt_test  = self.target[len_data:]

        return cndt_train, rslt_train, cndt_test, rslt_test

class _knnPCAPlt:
    def __init__(self,cndt_data,rslt_data):
        self.cndt_data = cndt_data
        self.rslt_data = rslt_data

    def dataPlot(self):
        cndt_data_ = StandardScaler().fit_transform(self.cndt_data)
        
        pca = PCA(n_components=2)
        pc = pca.fit_transform(cndt_data_)

        pc_y = np.c_[pc,self.rslt_data]
        df = pd.DataFrame(pc_y,columns=['PC1','PC2','diagnosis'])

        sns.scatterplot(data=df,x='PC1',y='PC2',hue='diagnosis')
        plt.show()
        
if __name__ == "__main__":
    try:
        cancer_data = load_breast_cancer()
        breast_cancer = _knnOneResult(cancer_data)
        cndt_train, rslt_train, cndt_test, rslt_test = breast_cancer.dataSetRatio(0.88)

        plot_data =_knnPCAPlt(cndt_train, rslt_train)
        plot_data.dataPlot()

        knn = KNN(n_neighbors =5)       # 가장 가까운 5가지를 선택한다.
        knn.fit(cndt_train,rslt_train)  # 데이터와 결과를 학습시킨다. 

        rslt_pred = knn.predict(cndt_test) # 데이터를 가지고 결과 값을 예상한다. 

        accuracy = np.sum(np.equal(rslt_pred,rslt_test))/len(rslt_test) # 실제 결과 값과 데이터를 구한다. 
        print(accuracy)

        plt.plot(1,1)
    except Exception:
        traceback.print_exc(file=sys.stdout)
