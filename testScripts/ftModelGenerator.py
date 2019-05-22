import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/MaxValueSynched1.csv")

print(dataset.head(2))
print (dataset.describe(include='all'))


x= dataset.iloc[:,1:200]
y= dataset.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(type(y_test[10]))