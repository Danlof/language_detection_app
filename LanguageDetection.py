import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.simplefilter("ignore")

import sklearn

print(sklearn.__version__)

# Loading the dataset
data = pd.read_csv("/media/danlof/dan files/data_science_codes/Project_3/deployment/Language_Detection.csv")
data.head()

# value count for each language
data["Language"].value_counts()

X = data["Text"]
y = data["Language"]

# converting categorical variables to numerical

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

le.classes_

## Text processing

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    text = text.lower()
    data_list.append(text)

## Train and test sets splitting

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

##Bag of words

# creating bag of words using countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(X_train)

x_train= cv.transform(X_train).toarray()
x_test = cv.transform(X_test).toarray()

## model creation and prediction

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train,y_train)

##prediction

y_pred = model.predict(x_test)

## Pipeline
from sklearn.pipeline import Pipeline

pipe = Pipeline([('vectorizer',cv),('multinomialNB',model)])
pipe.fit(X_train,y_train)

#Evaluating the model

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

# accuracy score
y_pred2 = pipe.predict(X_test)
ac2 = accuracy_score(y_test,y_pred2)
ac2

#2. Classification report
print(cr)

# confusion matrix

# visualising the confusion matrix
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)

# saving cv and the model

with open('model/trained_pipeline-0.1.0.pkl', 'wb') as f:
    pickle.dump(pipe,f)

# testing the model
text = 'Bella ciao'
y=pipe.predict([text])
le.classes_[y[0]],y