import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/MaxValueSynchedBalanced.csv")

print(dataset.head(2))

print(len(dataset.head(1)))

#print (dataset.describe(include='all'))


x= dataset.iloc[:,1:201]
y= dataset.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print("=====================")

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=200))
#Second  Hidden Layer
classifier.add(Dense(50, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)
eval_model

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)


#print("========================="+str(y_pred)+"==========================")

from sklearn.metrics import confusion_matrix
testevaluation = confusion_matrix(y_test, y_pred)
print(testevaluation)

a= testevaluation[0][0]
b = testevaluation[0][1]
c = testevaluation[1][0]
d = testevaluation[1][1]
print("final accuracy" + str((a+d)/(a+b+c+d)) )

print("===================================================")

print(y_test)
