# -*- coding: utf-8 -*-
"""
Created on Thu May 19 06:50:59 2022

@author: Orlando
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:00:38 2022

@author: Orlando
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import nltk
nltk.download('punkt')
from nltk import word_tokenize

import string

from utils import *

textdf = pd.read_csv("train_pos.csv")
textdf.head()


num_tokens=1000
seq_length = 100


mytokens = Tokenizer(num_words=num_tokens)
mytokens.fit_on_texts(textdf.text)

#prétraitements : 

encoded_docs = mytokens.texts_to_sequences(textdf.text)
#take out the ones that are less than seq_length
encoded_docs = [doc for doc in encoded_docs if len(doc)>seq_length]
print("number of sequences : ",len(encoded_docs))

padded_docs = pad_sequences(encoded_docs,maxlen=np.max([len(doc) for doc in encoded_docs]))
print("max length of sequences : ",np.max([len(p) for p in padded_docs]))
print("min length of sequences : ",np.min([len(p) for p in padded_docs]))
print("number of sequences : ",len(padded_docs))


#generating sequences qut we're just going to use one to set the network off
#it also helps defining lengths for the model

trainX = []
trainy = []
for doc in padded_docs:
  index = 0
  #first move forward till the whole sequence is information
  while index+seq_length<len(doc) and doc[index]==0:
    index+=1
  #from there begin creating sentences
  while index+seq_length<len(doc):
    trainX.append(doc[index:index+seq_length])
    trainy.append(doc[index+seq_length])
    index+=1
    if len(trainX)%100000==0:
      print(len(trainX)/1000,"thousand sequences added")
     
#sanity check
id = np.random.randint(0,len(trainX))
print("testing from",id,"th sequence : ")
for i in range(5):
  sequence = mytokens.sequences_to_texts(trainX[id:id+1])[0]
  #should find a way to display the word only once but the function takes multiple sequences
  toPredict = mytokens.sequences_to_texts(
      [[trainy[id],trainy[id]],[trainy[id],trainy[id]]])[0]
  print("...",sequence[len(sequence)-70:len(sequence)], "->", toPredict[0:int(len(toPredict)/2)])
  id+=1
  
#reshaping data
X = np.reshape(trainX,(len(trainX),seq_length,1))
X = X / float(num_tokens)
y=np_utils.to_categorical(trainy)
y = np.reshape(y,(len(y),len(y[0])))
print(X.shape)
print(y.shape)
        
#model :
model = Sequential()
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

#predict something so it can compile
pattern = list(trainX[0])
x = np.reshape(pattern, (1, len(pattern), 1))
x = x / float(num_tokens)
prediction = model.predict(x, verbose=0)

# load the network weights
filename = "weights-improvement-66-5.1545.hdf5"
model.load_weights(filename)
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#generate 500 sequences of 200 words each

int_to_word = dict([(i,w) for (w,i) in mytokens.word_index.items()])

start = np.random.randint(0, len(trainX)-1)
pattern = list(trainX[start])
print("Seed:")
print("\"", ' '.join([int_to_word[value] for value in pattern]), "\"")

sequence = []
# generate words
for i in range(500*200):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(num_tokens)
		prediction = model.predict(x, verbose=0)
		arglist = np.argsort(prediction)
		choice = np.random.randint(0,1)
		prediction = arglist[0][len(arglist[0])-1-choice]
		
		
		result = int_to_word[prediction+1]
		sequence.append(result)
		pattern.append(prediction)
		pattern = pattern[1:len(pattern)]
			
		if(i%200==199):
			print(i/200," sequences done")
    