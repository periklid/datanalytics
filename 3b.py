# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:29:38 2020

@author: Giannis
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import classification_report

movie_reviews = pd.read_csv("train_sentim.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape





from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


X = []
sentences = list(movie_reviews['Content'])
for sen in sentences:
    X.append(preprocess_text(sen))
    
y = movie_reviews['Label']

y = np.array(list( y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open(r'glove.6B.100d.txt/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))       

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

#score = model.evaluate(X_test, y_test, verbose=1)
# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

predictions = np.zeros(len(X), dtype=object)
lst = []
cols = ['Id', 'Predicted']
for i in range(len(X)-1):
    instance = X[i]
    
    instance = tokenizer.texts_to_sequences(instance)
    
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)
    
    flat_list = [flat_list]
    
    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    
    value = model.predict(instance)
   # print (value)
    
    for k in value:
        if k>=0.5  :
          
        
            predictions[i] =   1
        
        elif k <0.5 :
           # print (k,i)
            predictions[i] =   0
            
        curr_id = movie_reviews.iloc[i]['Id']
        lst.append([curr_id, predictions[i]])
        
        
        
pf = pd.DataFrame(lst, columns=cols)
pf.to_csv("sentiment_predictions_deep.csv", encoding="utf-8", index=False)
print( "accuracy", "     f1_score", "     precision", "    recall")
print( accuracy, f1_score, precision, recall)

