import urx 
import msvcrt
import cv2 as cv
import socket
import _thread

moveDist = 0.02
img_counter = 0 







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

            height =  float(msgArr[2])
            diameterX = float(msgArr[0])
            diameterY = float(msgArr[1])
            finalDiameter = max(diameterX,diameterY)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Rivet height:" + str(height))
            print("Rivet Diameter: " + str(finalDiameter))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
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





if __name__ == '__main__':
    
    global msgCollectFlag 
    msgCollectFlag= False
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_EXPOSURE, -120) 
    #print(type(robot.get_pose()))

    while True:

        ret,frame = cam.read()
        cv.imshow("test",frame)    

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
            
            get_sensor_data()
           
        elif(chr(keyInput & 255) == "d"):
            _thread.start_new_thread(moveRight, (moveDist,) )            
            #moveRight(0.03)
            print("moved right")

            _thread.start_new_thread(get_sensor_data,())
        elif (chr(keyInput & 255)== "s"):
            
            _thread.start_new_thread(moveDown, (moveDist,) )
            print("moved down")
            _thread.start_new_thread(get_sensor_data,())
        elif(chr(keyInput & 255)=="w"):
            
            _thread.start_new_thread(moveUp, (moveDist,) )
            print("moved up")
            _thread.start_new_thread(get_sensor_data,())

        elif(chr(keyInput & 255)=="i"):
            
            _thread.start_new_thread(moveIn, (moveDist,) )
            print("moved In")
            _thread.start_new_thread(get_sensor_data,())

        elif(chr(keyInput & 255)=="o"):
            
            _thread.start_new_thread(moveOut, (moveDist,) )
            print("moved Out")
            _thread.start_new_thread(get_sensor_data,())
          
        elif(chr(keyInput & 255) == "x"):
            _thread.start_new_thread(get_sensor_data,())
        
        elif(chr(keyInput & 255) == "c"):
            img_name = "img{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            img_counter += 1 
            print("picture taken")

        elif(chr(keyInput & 255) == "k"):
            _thread.start_new_thread(moveRobot,())

        elif(chr(keyInput & 255 ) == "b"):
            _thread.start_new_thread(moveRobotBack,())



            #code to take a snap shot and save in a specifivc folder 
