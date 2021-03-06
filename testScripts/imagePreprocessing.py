import cv2 as cv
import math
import os
import imutils


#ideal edge detection range  = 50-120

def dataPreProcessing(imgName):
    data = []
    img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/preProcessedImage/plate0/" + imgName,-1)
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

        #============================================================
    cv.imshow('image',gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("C:/Users/gulat/Desktop/grey.png",gray)


    #================================================================





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


def dataPreProcessing1(imgName):
    data = []
    img = cv.imread("C:/Users/gulat/Desktop/thesis/preProcessedImage/plate0/img8",-1)
    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    print("-------------------")
    img = img[(math.ceil(height/2)-50):(math.ceil(height/2)+50), (math.ceil(width/2)-60):(math.ceil(width/2)+30)]

    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    img =cv.resize(img,(math.ceil(width), math.ceil(height)))

    img = cv.blur(img,(5,5))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv.Canny(gray,50,120)

    #edges = cv.resize(edges,( math.ceil(100), math.ceil(100)))

    print("***************************************")
    height, width = edges.shape
    print(str(height)+":"+str(width))

    flatEdge = edges.ravel()
    data.append(flatEdge)
    print (len(data))
    
    return edges


if __name__ == '__main__':
    
    
    dataPreProcessing("img8.png")
    '''
    for i in range(0,280,1):
        
        
        if(i>=230 and i<280):
            image  =  dataPreProcessing("img"+str(i)+".png")
            #cv.imwrite("C:/Users/gulat/Desktop/plate88/img"+str(i)+".png",image)
            print("eeeeeeeeeeeeeee     "+str(i)+" rrrrrrrrrrrrrrrrrrr")
        
        #else:
        #    image  =  dataPreProcessing("img"+str(i)+".png")# cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/plate0/img7.png")
        #    cv.imwrite("C:/Users/gulat/Desktop/thesis/postProcessedImage/plate5/img"+str(i)+".png",image)

        #image  =  dataPreProcessing1("img"+str(i)+".png")
        #cv.imwrite("C:/Users/gulat/Desktop/thesis/postProcessedImage/plate5/img"+str(i)+".png",image)
    
    
    processed = dataPreProcessing1("img6.png")
    cv.imshow("processedImagess",processed)
    cv.waitKey(0)
    '''

   