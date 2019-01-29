import cv2 as cv
import os


if __name__ == '__main__':

    i=0
    for filename in os.listdir("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/zeroDegEvalGood/"):
        print(filename)
        img  = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/zeroDegEvalGood/"+str(filename))
        cv.imwrite("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/zeroDegEvalGood1/img"+str(i)+".png",img)
        print(str(i))
        i = i+1