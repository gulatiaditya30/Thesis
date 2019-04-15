import csv
import numpy as np
from sklearn import preprocessing
import os

'''  normalising and orrienting data row wise 
for j in range(0,1,1):  # plate no
    
    for i in range(0,280,1): # the rivet no
        if(i in [120,159,160,199,200,239,240,279,189] ): #missing rivets in 5-8 plates [120,121,160,161,200,201,240,241]
             continue
        z = np.loadtxt(open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(j)+"/forcelog"+str(i) +".csv", "rb"), delimiter=",")

        #print("The problem rivet :"+ str(i))

        fx = z[:,0]
        fy = z[:,1]
        fz = z[:,2]

        
        fx = preprocessing.normalize([fx])
        fxN = fx[0,:]

        fy = preprocessing.normalize([fy])
        fyN = fy[0,:]

        fz = preprocessing.normalize([fz])
        fzN = fz[0,4:]
        
        del z

        with open('C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_'+str(j)+'/normalised/forcelogN'+str(i)+'.csv','w',newline='') as fileOut:
            writer = csv.writer(fileOut)
            with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(j)+"/forcelog"+str(i)+".csv") as ftFile:        
                c=0
                fileReader = csv.reader(ftFile,delimiter=",")
                for row in fileReader:    
                    row.append(fxN[c])
                    row.append(fyN[c])
                    row.append(fzN[c])
                    writer.writerow(row)
                    c=c+1
                    if(c==499):
                        break
        
        ftFile.close()
        fileOut.close()
    print("++++++++++++++++++++++"+str(j)+"Plate done ++++++++++++++++++++++++++++++++++")
    
for j in range(0,9,1):                                    #putting every thing into one file
    for q,w,files in os.walk("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(j)+"/normalised"):
        for f in files:
            with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(j)+"/normalised/"+str(f)) as fileRead:
                newNormalise=[]
                fReader = csv.reader(fileRead, delimiter=',')

                for row in fReader:
                    avgg = (float(row[6])+float(row[7])+float(row[8]))/3
                    row.append(avgg)
                    newNormalise.append(row)
                
                with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_"+str(j)+"/newNormalised/"+str(f),"w",newline = '') as newNormalWrite:
                    for line in newNormalise:
                        writer= csv.writer(newNormalWrite)
                        writer.writerow(line)
            
            fileRead.close()
            newNormalWrite.close()

        print(" plate no : "+str(j)+" done ==========")
    '''

N=25
with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/ftDataColumwise.csv") as withoutAvg:
    with open("C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/columnwise/movingAvgFtDataColumwise.csv","a",newline='') as columnFtFile:
        withoutAvgReader = csv.reader(withoutAvg,delimiter=',')
        k=1
        for row in withoutAvgReader:
            row = list(map(float, row[1:len(row)-2]))
            print(" ----- k:"+ str(k))
            k=k+1
            avgRow = np.convolve(row, np.ones((N,))/N, mode='valid')
            writer= csv.writer(columnFtFile)
            writer.writerow(avgRow)
            
    columnFtFile.close()    
    withoutAvg.close()




        