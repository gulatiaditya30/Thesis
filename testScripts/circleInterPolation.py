import math
import matplotlib.pyplot as plt
import urx 
import socket
import _thread
import cv2 as cv

#always press k three times first to get intial points to interpolate the circle

class coordinate:
    x = 0
    y = 0

    def __init__(self, x, y):

        self.x = x
        self.y = y
    

def getPoint():
    global initialPoints
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    
    initialPoints.append(currentState.pos.x)
    initialPoints.append(currentState.pos.y)
    print(str(currentState.pos.x)+","+str(currentState.pos.y))
    robot.close()
    #return [currentState.pos.x,currentState.pos.y,currentState.pos.z]

def drawCircle(x1,y1,x2,y2,x3,y3):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    i=0
    circleInfo = circlePolation(x1,y1,x2,y2,x3,y3)
    circumPoints = circumfenceCoordinate(circleInfo[0],circleInfo[1],circleInfo[2])
    print (circumPoints[10].x)
    
    while(i<361):
        currentState.pos.x = circumPoints[i].x
        currentState.pos.y = circumPoints[i].y
        robot.set_pose(currentState,vel = 0.050,acc =2 )
        i=+1

    robot.close()



def circlePolation(x1,y1,x2,y2,x3,y3):

    a =  x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2

    b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (x1 - x2)

    c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2)

    x = -b/(2*a)
    y = -c/(2*a)

    return [x,y,math.hypot(x-x1,y-y1)]


def circumfenceCoordinate(Cx,Cy,r):
    i = 0
    circumCoordinates =[]

    xArr = []
    yArr = []

    while (i<361):
        

        x = ((r* math.cos(i)) + Cx)*0.001
        y = ((r*math.sin(i)) + Cy)*0.001
        p = coordinate(x,y)
        circumCoordinates.append(p)
        i+=1
    return circumCoordinates

if __name__ == "__main__":

    global initialPoints

    initialPoints =[]

    '''z = circlePolation(3,2,1,4,1,6)
    print(z)
    print("=================================")
    o = circumfenceCoordinate(z[0],z[1],z[2])
    xArr = [z[0]]
    yArr = [z[1]] 
    print(z[2])
    i=0
    while(i<361):
        #print("("+str(o[i].x)+"," +str(o[i].y)+")" )
        xArr.append(o[i].x)
        yArr.append(o[i].y)
        i+=1
    
    plt.plot(xArr, yArr, 'ro')
    plt.show()
    ================================================================================================================================='''
    cam = cv.VideoCapture(0)
    while True:
        ret,frame = cam.read()
        cv.imshow("test",frame)

        keyInput  =  cv.waitKey(1)

        if (chr(keyInput & 255) == "q"):
            exit()
        elif(chr(keyInput &255)=="p"):
            xArr = []
            yArr = []
            i=0
            z = circlePolation(initialPoints[0],initialPoints[1],initialPoints[2],initialPoints[3],initialPoints[4],initialPoints[5])
            o = circumfenceCoordinate(z[0],z[1],z[2])
            while(i<360):
                xArr.append(o[i].x)
                yArr.append(o[i].y)
                i+=1
            plt.plot(xArr,yArr,'ro')
            plt.show()

        elif (chr(keyInput & 255) == "k"):
            _thread.start_new_thread(getPoint, ())
        elif(chr(keyInput & 255 )=="d"):
            _thread.start_new_thread(drawCircle,(initialPoints[0],initialPoints[1],initialPoints[2],initialPoints[3],initialPoints[4],initialPoints[5])) 


