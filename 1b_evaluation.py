# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:10:27 2020

@author: Giannis
"""

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

pd.set_option('precision', 6)

# Read data #
train_data = pd.read_csv('train.csv',header=0, nrows=400)

# Drop useless columns #
#train_data = train_data.drop(['RowNum', 'Id', 'Title'], axis=1)

# 5-fold #
kf = StratifiedKFold(n_splits=5, random_state=123)

y_train = train_data["Label"]
X_train = train_data["Content"]

# Add labels #
le = preprocessing.LabelEncoder()
X_train_le = le.fit_transform(y_train)
X_train_cat = le.inverse_transform(X_train_le)

# Create matrix of TF-IDF features #
tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)



# Use LSA for dimensionality reduction #
svd = TruncatedSVD(n_components=100, random_state=123)

# Perform dimensionality reduction #
X_train_reduced = svd.fit_transform(X_train_tfidf)

# Choose classifiers to measure #

clf_RF = RandomForestClassifier(n_estimators=100, criterion="entropy", bootstrap=False, warm_start=True, random_state=123)
clf_SVC = SVC(C=5, kernel="linear", gamma="auto", random_state=123)
       # TruncatedSVD(n_components=2000, n_iter=5, random_state=42)

clfs = [ clf_RF, clf_SVC, clf_RF, clf_SVC ]
clfs_names = ["Random Forest SVD", "SVM SVD","Random Forest BoW", "SVM BoW"]

# Metrics #
scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

# Keep results #
prec_clfs, rec_clfs, f1_clfs, accu_clfs = ({} for i in range(4))

for i in range(len(clfs)):

    # Train without lsi #
    if i == 2 or i == 3:
        scores = cross_validate(clfs[i], X_train_tfidf, X_train_le, scoring=scoring, cv=kf)
    else:
        scores = cross_validate(clfs[i], X_train_reduced, X_train_le, scoring=scoring, cv=kf)

    # Save scores #
    prec_clfs[clfs_names[i]] = str(round(scores['test_precision_macro'].mean(), 6))
    rec_clfs[clfs_names[i]] = str(round(scores['test_recall_macro'].mean(), 6))
    f1_clfs[clfs_names[i]] = str(round(scores['test_f1_macro'].mean(), 6))
    accu_clfs[clfs_names[i]] = str(round(scores['test_accuracy'].mean(), 6))

# Add results in data frame and export a csv #
evaluation_metric_df = pd.DataFrame()
evaluation_metric_df.insert(loc=0, column='Statistic Measure', value=['Accuracy', 'Precision', 'Recall', 'F-Measure'])

# Insert results #
for i in range(len(clfs)):

    evaluation_metric_df.insert(loc=i + 1, column=clfs_names[i], value=[accu_clfs[clfs_names[i]],
                                                                        prec_clfs[clfs_names[i]],
                                                                        rec_clfs[clfs_names[i]],
                                                                        f1_clfs[clfs_names[i]]
                                                                        ])

# Create csv #
evaluation_metric_df.to_csv("EvaluationMetric_5fold.csv", encoding='utf-8', index=False, sep="\t")