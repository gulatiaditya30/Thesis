import urx 
import msvcrt
import cv2 as cv
import socket
import _thread
import csv
import math
import time

moveDist = 0.01
img_counter = 0
plateCounter = 0
exposureV = -2
# exposure should be set to -5 with a distance of camera to plate of 13.5 cm 
# do take images of all positions even if there is no rivet in the hole , it will help maintain the labelling 
#if the script crahes and you have to again take images please change the img_counter in line 10 variable to last img_counter +1 
# when collecting data for the new plate please update the plate folder no in line 183 and 254 
# similarly for new plate  data sensor readings update the csv file name in line 123 not important while collecting images



def moveLeft(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y -= dist
    
    robot.set_pose(currentState,vel=5)
    robot.close()

def moveRight(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y += dist 
    robot.set_pose(currentState,vel=5)
    robot.close()

def moveUp(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.z += dist
    robot.set_pose(currentState,vel=5)
    robot.close()
    

def moveDown(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.z -= dist
    robot.set_pose(currentState,vel=5) 
    robot.close()

def moveIn(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.x -= dist
    robot.set_pose(currentState,vel = 5)
    robot.close()

def moveOut(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.x += dist
    robot.set_pose(currentState,vel = 5)
    robot.close()

def rotate():
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.orient.rotate_xb(90/2)
    robot.set_pose(currentState,vel = 5)
    robot.close()

def get_sensor_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("160.69.69.110",8190))

    msg  = s.recv(1028)
    msg = msg.decode("utf-8")
    s.close()
    print("========================================================")
    print(msg)
    print("=======================================")

def findRightReading(robo):
    global msgCollectFlag
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(("160.69.69.110",8190))
    s.send(b"Start#")
    msgArr = []
    print(msgCollectFlag)
    finalDiameter = 0.0
    diameterX = 0.0
    diameterY = 0.0
    
    while(msgCollectFlag):
        msg = s.recv(1028)
        msg = msg.decode("utf-8")
        msg = msg.split("#")[0]
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        print(msg)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        msgArr = msg.split(",")
        if ((msg != "OK") and (len(msgArr) == 3)):
            if(msgArr[2]!="INVALID"):
                height =  float(msgArr[2])
            else:
                height = msgArr[2]
            
            if(msgArr[0]!="INVALID"):
                diameterX = float(msgArr[0])
            else:
                diameterX = msgArr[0]

            if(msgArr[1]!="INVALID"):
                diameterY = float(msgArr[1])
            else:
                diameterY = msgArr[1]
            #finalDiameter = max(diameterX,diameterY)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Rivet height:" + str(height))
            print("Rivet DiameterX: " + str(diameterX))
            print("Rivet DiameterY: " + str(diameterY))
            row = [str(height),str(diameterX),str(diameterY)]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            with open('sensorData/dataVideo.csv','a',newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
        #if (decD == "1"):
        #    msgArr.append(float(valD))
    s.send(b"Stop#")
    s.close()
    robo.close
    #print("*******************************************")
    #print(max(msgArr))
    #print("**************************************")




def moveRobot():

    global msgCollectFlag
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    #currentState.pos.y -= 0.075
    #robot.set_pose(currentState,vel = 5)
    currentState.pos.y += 0.02
    msgCollectFlag = True
    _thread.start_new_thread(findRightReading, (robot,))
    robot.set_pose(currentState,vel = 0.005,acc =2 )
    msgCollectFlag =False

def moveRobotBack():
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y -= 0.02
    robot.set_pose(currentState,vel = 0.005,acc =2 )

def imgPreProcessing(imgAdd,counter):
    data = []
    img = cv.imread(imgAdd,-1)#'C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/img0.png',-1)
    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    print("-------------------")
    img = img[(math.ceil(height/2)-100):(math.ceil(height/2)+100), (math.ceil(width/2)-100):(math.ceil(width/2)+100)]

    height, width ,depth= img.shape
    print(str(height)+":"+str(width))
    #img =cv.resize(img,(math.ceil(width), math.ceil(height)))

    img = cv.blur(img,(3,3))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv.Canny(gray,100,150)

    #img = cv.resize(img,( math.ceil(640), math.ceil(6)))

    flatEdge = edges.ravel()
    data.append(flatEdge)
    

    cv.imwrite("C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/postProcessedImage/plate"+str(plateCounter)+"/img"+str(counter)+".png", edges)
    





if __name__ == '__main__':
    
    global msgCollectFlag 
    msgCollectFlag= False
    cam = cv.VideoCapture(0)
    cam.set(15,-5) #range is from -1 to -13 from long exposure to short exposure
    #print(type(robot.get_pose()))

    while True:

        ret,frame = cam.read()
        cv.imshow("test",frame,)    

        keyInput  =  cv.waitKey(1)

        #keyInput = msvcrt.getch()
        #if(keyInput.decode("utf-8") == "q"):
        #    exit()
        #print(type(keyInput))
        #print(keyInput)
        
        if(chr(keyInput & 255) == "q"):
            exit()
        
        elif(chr(keyInput & 255)=="a"):
            _thread.start_new_thread(moveLeft, (moveDist,) )
            #moveLeft(0.03)
            print("moved Left")
            
            #get_sensor_data()
           
        elif(chr(keyInput & 255) == "d"):
            _thread.start_new_thread(moveRight, (moveDist,) )            
            #moveRight(0.03)
            print("moved right")

            #_thread.start_new_thread(get_sensor_data,())
        elif (chr(keyInput & 255)== "s"):
            
            _thread.start_new_thread(moveDown, (moveDist,) )
            print("moved down")
            #_thread.start_new_thread(get_sensor_data,())
        elif(chr(keyInput & 255)=="w"):
            
            _thread.start_new_thread(moveUp, (moveDist,) )
            print("moved up")
            #_thread.start_new_thread(get_sensor_data,())

        elif(chr(keyInput & 255)=="i"):
            
            _thread.start_new_thread(moveIn, (moveDist,) )
            print("moved In")
            #_thread.start_new_thread(get_sensor_data,())

        elif(chr(keyInput & 255)=="o"):
            
            _thread.start_new_thread(moveOut, (moveDist,) )
            print("moved Out")
            #_thread.start_new_thread(get_sensor_data,())
          
        elif(chr(keyInput & 255) == "x"):
            _thread.start_new_thread(get_sensor_data,())
        
        elif(chr(keyInput & 255) == "c"):
            img_name = "C:/Users/gulat/Desktop/thesis/gitThesis/testScripts/preProcessedImage/plate"+str(plateCounter)+"/img"+str(img_counter)+".png"
            cv.imwrite(img_name, frame)
            imgPreProcessing(img_name,img_counter)

            img_counter += 1 
            print("picture taken")

        elif(chr(keyInput & 255) == "k"):
            _thread.start_new_thread(moveRobot,())

        elif(chr(keyInput & 255) == "l"):
            for l in range(0,40):
                _thread.start_new_thread(moveRobot,())
                time.sleep(15)
            print("********************************************** DONE ***************************************************************")

        elif(chr(keyInput & 255 ) == "b"):
            _thread.start_new_thread(moveRobotBack,())



            #code to take a snap shot and save in a specifivc folder 
