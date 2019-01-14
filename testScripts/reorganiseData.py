from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
import cv2 as cv


def validityCheck(plateNo,rivetNo):
    with open('C:/Users/gulat/Desktop/thesis/gitThesis/images/labeledPlate.csv')as labbeledFile:
        fileRead = csv.reader(labbeledFile,delimiter = ',')
        x = list(fileRead)
        if(x[rivetNo][plateNo+1] == 'NA'):
            return 'NA'
        else:
            return str(x[rivetNo][plateNo+1])



def main1():

  # Load training and eval data
  train_data = []
  eval_data = []

  train_labels = []
  eval_labels =[]
  bNo = 0
  gNo = 0
  for z in range (0,181,30):
    for x in range(0,9,1):         #plates
          for y in range(0,280,1):
            if(validityCheck(x,y) != 'NA' ):
                if(validityCheck(x,y)=='0'):
                    img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/postProcessedImage/degree"+str(z)+"/plate"+str(x)+"/img"+str(y)+".png")
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    cv.imwrite("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/bad/img"+str(bNo)+".png",img)
                    bNo = bNo + 1
                elif((validityCheck(x,y))=='1'):
                    img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/postProcessedImage/degree"+str(z)+"/plate"+str(x)+"/img"+str(y)+".png")
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    cv.imwrite("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/good/img"+str(gNo)+".png",img)
                    gNo = gNo + 1
    print("good: " +str(gNo) + " bad: "+str(bNo) )
if __name__ == "__main__":

    main1()