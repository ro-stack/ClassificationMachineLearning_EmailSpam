# Ronan Hayes - CI213 
import numpy as np
import pandas as pd 

import sklearn   
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate  

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('dataset.csv').values
np.random.shuffle(dataset)

X = dataset[ : , :48 ]
Y = dataset[ : , -1 ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .33, random_state = 17)

# Bernoulli Naive Bayes
BernNB = BernoulliNB()
BernNB.fit(X_train, Y_train)
y_expect = Y_test
y_pred = BernNB.predict(X_test)   
print ("Bernoulli Accuracy Score: ")
BernNB_accuracy = (accuracy_score(y_expect, y_pred))
print (BernNB_accuracy * 100, "%")
print ("Bernoulli Confusion Matrix: ")
print(confusion_matrix(y_expect, y_pred))
BernNB = cross_validate(BernNB, X, Y, cv = 10)
print("Bernoulli Cross Validation Results:")
cross_val_results = BernNB['test_score']
print(cross_val_results)
print("Bernoulli Cross Validation Mean Score:")
print(cross_val_results.mean() * 100, "%")
print ("")

# Multinomial Naive Bayes
MultiNB = MultinomialNB()
MultiNB.fit(X_train, Y_train)
y_pred = MultiNB.predict(X_test)
print ("Multinomial Accuracy Score: ")
MultiNB_accuracy = (accuracy_score(y_expect, y_pred))
print (MultiNB_accuracy * 100, "%")
print ("Multinomial Confusion Matrix: ")
print(confusion_matrix(y_expect, y_pred))
MultiNB = cross_validate(MultiNB, X, Y, cv = 10)
print("Multinomial Cross Validation Results:")
cross_val_results = MultiNB['test_score']
print(cross_val_results)
print("Multinomial Cross Validation Mean Score:")
print(cross_val_results.mean() * 100, "%")
print ("")

# Gaussian Naive Bayes
GausNB = GaussianNB()
GausNB.fit(X_train, Y_train)
y_pred = GausNB.predict(X_test)
print ("Gaussian Accuracy Score: ")
GausNB_accuracy = (accuracy_score(y_expect, y_pred))
print (GausNB_accuracy * 100, "%")
print ("Gaussian Confusion Matrix: ")
print(confusion_matrix(y_expect, y_pred))
GausNB = cross_validate(GausNB, X, Y, cv = 10)
print("Gaussian Cross Validation Results:")
cross_val_results = GausNB['test_score']
print(cross_val_results)
print("Gaussian Cross Validation Mean Score:")
print(cross_val_results.mean() * 100, "%")
print ("")

# Bernoulli (Binarized) Naive Bayes
BernNB_Binarized = BernoulliNB(binarize = 0.1)
BernNB_Binarized.fit(X_train, Y_train)
y_expect = Y_test
y_pred = BernNB_Binarized.predict(X_test)   
print ("Bernoulli 'Binarized' Accuracy Score: ")
BernNB_Binarized_accuracy = (accuracy_score(y_expect, y_pred))
print (BernNB_Binarized_accuracy * 100, "%")
print ("Bernoulli 'Binarized' Confusion Matrix: ")
print(confusion_matrix(y_expect, y_pred))
BernNB_Binarized = cross_validate(BernNB_Binarized, X, Y, cv = 10)
print("Bernoulli 'Binarized' Cross Validation Results:")
cross_val_results = BernNB_Binarized['test_score']
print(cross_val_results)
print("Bernoulli 'Binarized' Cross Validation Mean Score:")
print(cross_val_results.mean() * 100, "%")
