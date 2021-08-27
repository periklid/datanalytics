# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:31:24 2020

@author: Giannis
"""

import pandas as pd
import string
import re
import collections
import nltk
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("train_sentim.csv");
test_data = pd.read_csv("test_without_labels_sentim.csv")

string.punctuation

ps = nltk.PorterStemmer()

def remove_nonalphanumeric(text):
    text_out =  "".join([char for char in text if char not in string.punctuation])
    return text_out

def tokanization(text):
    tokens = re.split('\W+',text)
    return tokens

def stemming(text):
    out_text = [ps.stem(word) for word in text]
    return out_text


def remove_stopwords(torkanized_list):
    text_out = [word for word in torkanized_list if word not in stopword]
    return text_out

train_data['nopunct'] = train_data['Content'].apply(lambda x: remove_nonalphanumeric(x))
    
train_data['tokans'] = train_data['nopunct'].apply(lambda x:tokanization(x.lower()))

train_data['FirstDataSetStem'] = train_data['tokans'].apply(lambda x: stemming(x))

train_data['FirstDataSetStem'].head()

#nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')

train_data['SecondDataSet'] = train_data['FirstDataSetStem'].apply(lambda x: remove_stopwords(x))

train_data['SecondDataSet'][:10]

SecondDataSet = list(itertools.chain.from_iterable(train_data['SecondDataSet']))

flat_list = list(itertools.chain.from_iterable(train_data['SecondDataSet']))

fd = nltk.FreqDist(flat_list)

word_toKeep = list(filter(lambda x: 2000>x[1]>3,fd.items()))

word_list_ToKeep = [item[0] for item in word_toKeep]

def remove_lessfreq(tokanized_train_data):
    text_out = [word for word in tokanized_train_data if word in word_list_ToKeep]
    return text_out

train_data['ThirdDataSet'] = train_data['SecondDataSet'].apply(lambda x: remove_lessfreq(x))

train_data.head()

ListofUniqueWords = set(list(itertools.chain.from_iterable(train_data['ThirdDataSet'])))

def join_tokens(tokens):
    document = " ".join([word for word in tokens if not word.isdigit()])
    return document

train_data['ThirdDataSet_Docs'] = train_data['ThirdDataSet'].apply(lambda x: join_tokens(x))


cv = CountVectorizer(ListofUniqueWords)

countvector = cv.fit_transform(train_data['ThirdDataSet_Docs'])
countvectorDF = pd.DataFrame(countvector.toarray())
countvectorDF.columns = cv.get_feature_names()

print(countvectorDF[:10])

X = countvectorDF
y = train_data['Label']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


logreg = LogisticRegression()

logreg = logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)


cols = ['Id', 'Predicted']
lst = []

# Lst: list of lists #
# Every single list  #
# contains id and    #
# predicted category #

for i in range(800):
    curr_id = train_data.iloc[i]['Id']
    lst.append([curr_id, y_pred[i]])


pf = pd.DataFrame(lst, columns=cols)
pf.to_csv("sentiment predictions.csv", encoding="utf-8", index=False)

from sklearn.metrics import classification_report

import sklearn.metrics as metrics
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

