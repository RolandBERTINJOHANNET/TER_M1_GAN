# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:41:31 2022

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


#lecture des données
textdf = pd.read_csv("IMDB Dataset.csv")
textdf.head()

#récupération de 100 de chaque classe et concaténation dans une liste
groups = textdf.groupby("sentiment")
pos = groups.get_group("positive").sample(500)
neg = groups.get_group("negative").sample(500)
X = np.vstack([neg,pos])
y = np.hstack([[0]*500,[1]*500])


#mise au bon format
X = np.array([x[0] for x in X])


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
X=tv.fit_transform(X)

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,shuffle=True,test_size=0.3,train_size=0.7)

clf = LogisticRegression()
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print("accuracy : ",classification_report(y_test, pred,output_dict=True)['accuracy'])


#PCA
pca = PCA(n_components = 2)
components = pca.fit_transform(X.toarray())

plt.scatter([cp[0] for cp in components[0:200]],[cp[1] for cp in components[0:200]],color="red",label='negative')
plt.scatter([cp[0] for cp in components[500:700]],[cp[1] for cp in components[500:700]],color="green",label='positive')
plt.legend()