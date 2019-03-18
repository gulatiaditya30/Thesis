import csv
import numpy as np
from sklearn import preprocessing


'''
with open("address.csv") as ftFile:

    fileReader = csv.reader(ftFile,delimiter=",")

    for row in fileReader:
        fx = row[0]
'''
for j in range(8,9,1):  # the plate 
    
    for i in range(0,280,1): # the rivet no
        if(i in [120,121,160,161,200,201,240,241]): #missing rivets in 0-4 plates [120,159,160,199,200,239,240,279]
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
        fzN = fz[0,:]
        
        del z

        with open('C:/Users/gulat/Desktop/thesis/gitThesis/trainingData/ftData/ft_plate_'+str(j)+'/normalised/forcelogN'+str(i)+'.csv','a',newline='') as fileOut:
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

        