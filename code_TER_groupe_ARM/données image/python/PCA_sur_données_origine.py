# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:22:03 2022

@author: Orlando
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()

print("train", x_train.shape, y_train.shape)
print("test", x_test.shape, y_test.shape)

def load_012():
  (x_train,y_train), (_, _) = load_data()
  zz=np.zeros(len(y_train))
  idxz = np.where(y_train==zz)
  oo=np.ones(len(y_train))
  idxo = np.where(y_train==oo)
  dd=2*np.ones(len(y_train))
  idxd = np.where(y_train==dd)

  index = np.hstack([idxz[0],idxo[0],idxd[0]])
  sub_train = x_train[index]
  return sub_train,y_train[index]

#ranger les 0,1,2 sous une forme acceptable pour un classificateur :
X, y=load_012()
X = np.array([x.flatten() for x in X])

#train test split :
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=(True))

#classification

clf = LogisticRegression()
clf=clf.fit(x_train, y_train)
pred = clf.predict(x_test)
cl_rep = classification_report(y_test,pred,output_dict=True)
print("accuracy sur les données d'origine : ",cl_rep['accuracy'])

#faire une PCA :
    
def load_2():
  (x_train,y_train), (_, _) = load_data()
  dd=2*np.ones(len(y_train))
  idxd = np.where(y_train==dd)
  sub_train = x_train[idxd[0]]
  return sub_train,y_train[idxd[0]]

def load_01():
  (x_train,y_train), (_, _) = load_data()
  zz=np.zeros(len(y_train))
  idxz = np.where(y_train==zz)
  oo=np.ones(len(y_train))
  idxo = np.where(y_train==oo)

  index = np.hstack([idxz[0],idxo[0]])
  sub_train = x_train[index]
  return sub_train,y_train[index]

#charger des 0 et 1, et des 2, les décoller, et faire une PCA.
X_01,y_01 = load_01()
X_01=np.reshape(X_01,(X_01.shape[0],28*28))
X_0=np.reshape([X_01[i] for i in range(len(X_01)) if y_01[i]==0][0:200], (200,28*28))
y_0=[0]*200
y_1=[1]*200
X_1=np.reshape([X_01[i] for i in range(len(X_01)) if y_01[i]==1][0:200], (200,28*28))
X_2,y_2 = load_2()
X_2=np.reshape(X_2[0:200],(200,28*28))

#group together
X=np.vstack([X_0,X_1,X_2])
y=np.hstack([y_0,y_1,y_2])


#PCA
pca = PCA(n_components = 2)
components = pca.fit_transform(X)

plt.scatter([cp[0] for cp in components[0:200]],[cp[1] for cp in components[0:200]],color="red",label='0')
plt.scatter([cp[0] for cp in components[200:400]],[cp[1] for cp in components[200:400]],color="green",label='1')
plt.scatter([cp[0] for cp in components[400:600]],[cp[1] for cp in components[400:600]],color="blue",label='2')
plt.legend()