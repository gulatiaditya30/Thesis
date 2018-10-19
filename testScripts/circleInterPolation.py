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

def moveAroundCircle(x1,y1,x2,y2,x3,y3):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    i=0
    circleInfo = circlePolation(x1,y1,x2,y2,x3,y3)
    circumPoints = circumfenceCoordinate(circleInfo[0],circleInfo[1],circleInfo[2])
    print("@@@@@@@@@@@@@@@@@@@@@@@@*********************************@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print (len(circumPoints))
    
    while(i<360):
        currentState = robot.get_pose()
        currentState.pos.x = circumPoints[0][i]
        currentState.pos.y = circumPoints[1][i]
        print(str(circumPoints[0][i]) +","+str(circumPoints[1][i]))
        robot.set_pose(currentState,vel = 0.050,acc =2 )
        i=i+1

    robot.close()



def circlePolation(x1,y1,x2,y2,x3,y3):

    a =  x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2

    b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1)

    c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2)

    x = -b/(2*a)
    y = -c/(2*a)

    return [x,y,math.hypot(x-x1,y-y1)]


def circumfenceCoordinate(Cx,Cy,r):
    i = 0
    circumCoordinates =[]

    xArr = []
    yArr = []

    while (i<95):
        

        x = ((r* math.cos(i)) + Cx)#*0.001
        y = ((r*math.sin(i)) + Cy)#*0.001
        xArr.append(x)
        yArr.append(y)
        p = coordinate(x,y)
        circumCoordinates.append(p)
        i+=1
    return [xArr,yArr]

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
            print(len(initialPoints))
            if (len(initialPoints)<6):
                print("select atleast 3 points on the circumfence")
            else:
                xArr = []
                yArr = []
                i=0
                z = circlePolation(initialPoints[0],initialPoints[1],initialPoints[2],initialPoints[3],initialPoints[4],initialPoints[5])
                o = circumfenceCoordinate(z[0],z[1],z[2])
                while(i<90):
                    xArr.append(o[0][i])
                    yArr.append(o[1][i])
                    i+=1
                plt.plot(xArr,yArr,'ro')
                iniX = [z[0],initialPoints[0],initialPoints[2],initialPoints[4]]
                iniY = [z[1],initialPoints[1],initialPoints[3],initialPoints[5]]
                plt.plot(iniX,iniY,'go')
                plt.show()

        elif (chr(keyInput & 255) == "k"):
            _thread.start_new_thread(getPoint, ())


        elif(chr(keyInput & 255 )=="d"):
            if (len(initialPoints)<6):
                print("select atleast 3 points on the circumfence")
            else:
                _thread.start_new_thread(moveAroundCircle,(initialPoints[0],initialPoints[1],initialPoints[2],initialPoints[3],initialPoints[4],initialPoints[5])) 


