import cv2 as cv
import math
import os
import imutils

def dataPreProcessing(imgName):
    data = []
    img = cv.imread("C:/Users/gulat/Desktop/thesis/preProcessedImage/plate8/" + imgName,-1)
    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    print("-------------------")
    img = img[(math.ceil(height/2)-50):(math.ceil(height/2)+50), (math.ceil(width/2)-50):(math.ceil(width/2)+40)]

    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    img =cv.resize(img,(math.ceil(width), math.ceil(height)))

    img = cv.blur(img,(5,5))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv.Canny(gray,100,150)

    #edges = cv.resize(edges,( math.ceil(100), math.ceil(100)))

    print("***************************************")
    height, width = edges.shape
    print(str(height)+":"+str(width))

    flatEdge = edges.ravel()
    data.append(flatEdge)
    print (len(data))

    #cv.imwrite("C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/plate0/" + imgName  , edges)
    return edges


if __name__ == '__main__':
    #noImages = len(os.listdir("C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/preProcessedImage/plate0"))
    #print(noImages)
    
    for i in range(0,280,1):
        image  =  dataPreProcessing("img"+str(i)+".png")# cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/plate0/img7.png")
        cv.imwrite("C:/Users/gulat/Desktop/thesis/postProcessedImage/plate8/img"+str(i)+".png",image)
    '''
    for angle in range (0,360,15):
        rotated = imutils.rotate(image, angle)
        cv.imshow("rotate try",rotated)
        cv.waitKey(0)
    '''    

   