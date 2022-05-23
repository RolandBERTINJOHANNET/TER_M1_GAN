# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:13:33 2022

@author: Orlando
"""


import numpy as np
import pandas as pd


import nltk
nltk.download('punkt')
from nltk import word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords
stop = list(stopwords.words("english"))

from sklearn.model_selection import train_test_split

import string

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#lecture des données origine
textdf = pd.read_csv("IMDB Dataset.csv")
textdf.head()


#récupération de 500 pos, 500 neg,, et concaténation dans une liste
groups = textdf.groupby("sentiment")
pos = groups.get_group("positive").sample(500)
neg = groups.get_group("negative").sample(500)
fakedf = pd.read_csv("texte_genere.csv")["teste généré"]
X = np.vstack([neg,pos])
y = np.hstack([[0]*500,[1]*500])


#mise au bon format et ajout des fakes
X = np.array([x[0] for x in X])
X = np.hstack([X,fakedf])


#preprocessing du texte :
newDocs = []
for i in range(len(X)):
    doc = X[i]
	#suppression des br
    doc = " ".join([word for word in doc.split(" ") if not 
					("<br />" in word or "<br"in word or "/>"in word or "/><br" in word)])
    #suppression de la ponctuation :
    new_doc = doc.translate(str.maketrans('','',string.punctuation))
    #suppression des stop words
    new_doc = " ".join([word for word in new_doc.split(" ") if word not in stop])
    #suppression des nombres
    new_doc = "".join([word for word in new_doc if not word.isdigit()])
	#suppression des br
	#plus tard
	
    newDocs.append(new_doc)

#tfidf
tv = TfidfVectorizer(lowercase=False, max_features=2000)
X=tv.fit_transform(X).toarray()

#préparation pour l'expe : il faut une liste de 500 neg, une de 500 pos fake et 500 pos real
X_neg = X[0:500]
X_pos_real = X[500:1000]
X_pos_fake = X[1000:1500]

#expérimentation : 

#dans la boucle
accuracies = []
for pourcentage in np.arange(0.05,0.951,0.1):
    #-construire la liste de deux selon les pourcentages
    X_pos = np.vstack([X_pos_fake[0:int(pourcentage*500)],X_pos_real[int(pourcentage*500):500]])
    if len(X_pos) <500:
        X_pos = np.vstack([X_pos,X_pos_fake[190]])    
    #-la concaténer aux négatifs
    X_expe = np.vstack([X_neg,X_pos_real])
    y_expe = [0]*500 + [1]*500
	
    accstock=[]
    for i in range(5):
        #-entraîner LR dessus
        clf = LogisticRegression()
        x_train,x_test,y_train,y_test=train_test_split(X_expe,y_expe,shuffle=True)
        clf.fit(x_train,y_train)
        a_predire=[x_test[x] for x in range(len(x_test)) if y_test[x]==1]
        pred = clf.predict(a_predire)
        clreport = classification_report([1]*len(pred), pred,output_dict=True)
        accstock.append(clreport['accuracy'])
    accuracies.append(np.mean(accstock))









#train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,shuffle=True,test_size=0.3,train_size=0.7)

clf = LogisticRegression()
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print("accuracy : ",classification_report(y_test, pred,output_dict=True)['accuracy'])


#PCA
pca = PCA(n_components = 2)
components = pca.fit_transform(X)

plt.scatter([cp[0] for cp in components[0:200]],[cp[1] for cp in components[0:200]],color=(1,0,0,0.65),label='negative base de données 2')
plt.scatter([cp[0] for cp in components[500:700]],[cp[1] for cp in components[500:700]],color=(0,1,0,0.65),label='positive base de données 2')
plt.scatter([cp[0] for cp in components[1000:1200]],[cp[1] for cp in components[1000:1200]],color=(0,0,1,0.65),label='generated positive')
plt.legend()