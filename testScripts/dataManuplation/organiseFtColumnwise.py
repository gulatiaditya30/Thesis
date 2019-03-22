import csv
import numpy as np
from sklearn import preprocessing
import os

'''
finalGood=[]
finalBad=[]

plateNo=0

for plateNo in range(0,9,1):
    
    with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/laserSensorData/data"+str(plateNo)+".csv") as labelFile:
        labelFileReader = csv.reader(labelFile,delimiter = ",")
        labelsValue = list(labelFileReader)
            
        for q,w,files in os.walk("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(plateNo)+"/normalised"):
            goodC=0
            for f in files:
                k=f.split("N")
                k=k[1].split(".")
                if(plateNo ==1 and int(k[0])>240):
                    continue
                if(labelsValue[int(k[0])][6]=="TRUE"):
                    goodC=goodC+1
                    finalGood.append(['C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_'+str(plateNo)+'/newNormalised/'+str(f)])
                elif(labelsValue[int(k[0])][6]=="FALSE"):
                    finalBad.append(["C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(plateNo)+"/newNormalised/"+str(f)])
            
        print(" plate no : "+str(plateNo)+ " done =============")

print(len(finalGood))
with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/good/goodAdress.csv","w",newline = '') as goodWrite:

    for line in finalGood:
        writer= csv.writer(goodWrite)
        writer.writerow(line)

with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/bad/badAdress.csv","w",newline = '') as badWrite:

    for line in finalBad:
        writer= csv.writer(badWrite)
        writer.writerow(line)
goodWrite.close() 
badWrite.close()   
'''

columnft = []
with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/bad/badAdress.csv") as badAdress:
    badAddressReader = csv.reader(badAdress,delimiter = ",")
    
    i = 0
    for row in badAddressReader:
        rivetcolumnft = []
        with open(str(row[0])) as tempRivetCSV:
            rivetcolumnft.append(0)
            tempRivetReader = csv.reader(tempRivetCSV,delimiter=",")
            for k in tempRivetReader:
                rivetcolumnft.append(k[9])
        columnft.append(rivetcolumnft)
        #print("rivet"+str(i) +"done")
        i=i+1

with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/good/goodAdress.csv") as goodAdress:
    goodAddressReader = csv.reader(goodAdress,delimiter = ",")
    i = 0
    for row in goodAddressReader:
        rivetcolumnft = []
        with open(str(row[0])) as tempRivetCSV:
            rivetcolumnft.append(1)
            tempRivetReader = csv.reader(tempRivetCSV,delimiter=",")
            for k in tempRivetReader:
                rivetcolumnft.append(k[9])
        columnft.append(rivetcolumnft)
        #print("rivet"+str(i) +"done")
        i=i+1






    with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/ftDataColumwise.csv","w",newline='') as columnFtFile:
        print("========================== WRITING INTO COLUMN FILE ============================")
        for line in columnft:
            writer= csv.writer(columnFtFile)
            writer.writerow(line)

    



