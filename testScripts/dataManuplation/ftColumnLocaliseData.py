import csv
import numpy as np
from sklearn import preprocessing
import os

with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/movingAvgFtDataColumwise.csv") as goodAdress:
    goodAddressReader = csv.reader(goodAdress,delimiter = ",")
    i = 0
    newRow=[]
    for row in goodAddressReader:
        print("max for row "+str(i)+": "+str(row.index(max(row))))
        i=i+1
        newRow=[]


    goodAdress.close()

    '''
    with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/ftDataColumwise.csv","w",newline='') as columnFtFile:
        print("========================== WRITING INTO COLUMN FILE ============================")
        for line in columnft:
            writer= csv.writer(columnFtFile)
            writer.writerow(line)

    '''



