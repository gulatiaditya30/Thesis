import cv2 as cv
import math

def dataPreProcessing():
    data = []
    img = cv.imread('C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/img0.png',-1)
    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    print("-------------------")
    img = img[(math.ceil(height/2)-100):(math.ceil(height/2)+100), (math.ceil(width/2)-100):(math.ceil(width/2)+100)]

    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    img =cv.resize(img,(math.ceil(width), math.ceil(height)))

    img = cv.blur(img,(5,5))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv.Canny(gray,100,150)

    #img = cv.resize(img,( math.ceil(640), math.ceil(6)))

    flatEdge = edges.ravel()
    data.append(flatEdge)
    print (len(data))

    cv.imwrite("im.png", edges)
    #return img


if __name__ == '__main__':

    dataPreProcessing()