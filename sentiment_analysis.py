import ast
import keras
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
from keras.layers import  Activation
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
import sys
import keras.backend as K

BASE_DIR = '/home/temp/Desktop/'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
batch_size=32

summary=[]
text=[]
rating=[]
labels=[]
x_test=[]
x_train=[]
y_test=[]
y_train=[]

#preprocessing data

def pre_processing():
	f=open("pos_amazon_cell_phone_reviews.json",'r')
	line=f.readline()
	line=ast.literal_eval(line)
	list1=line['root']
	for i in list1:
		d=i
		summary.append(d['summary'])
		text.append(d['text'])
		rating.append(d['rating'])
		labels.append(1)


pre_processing()




def pre_processing():
	f=open("neg_amazon_cell_phone_reviews.json",'r')
	line=f.readline()
	line=ast.literal_eval(line)
	list1=line['root']
	for i in list1:
		d=i
		summary.append(d['summary'])
		text.append(d['text'])
		rating.append(d['rating'])
		labels.append(0)
	

pre_processing()

#tokenize

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(summary)
sequences = tokenizer.texts_to_sequences(summary)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)




#preparing the embedding layer

embeddings_index = {}
f = open(os.path.join(BASE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#Building model

print('Build model...')
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))



#precision and recall

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

#compiling

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',precision,recall])

#training

print('Train...')


for i in range(0,3000):
    temp=random.randint(0,len(data))
    x_train.append(list(data[temp]))
    y_train.append(labels[temp])


for i in range(0,500):
    temp=random.randint(0,len(data))
    x_test.append( list(data[temp]))
    y_test.append(labels[temp])




model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2,
          validation_data=(x_test, y_test))
score, acc,precision,recall = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Score:', score)
print('Accuracy:', acc)
print('Precision',precision)
print('Recall',recall)




		
	
	





