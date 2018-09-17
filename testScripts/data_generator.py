import urx 
import msvcrt
import cv2 as cv
import socket
import _thread

stepSize = 0.03
img_counter = 0 





def moveLeft(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y -= dist
    robot.set_pose(currentState)
    robot.close()

def moveRight(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y += dist 
    robot.set_pose(currentState)
    robot.close()

def moveUp(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.z += dist
    robot.set_pose(currentState)
    robot.close()
    

def moveDown(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.z -= dist
    robot.set_pose(currentState) 
    robot.close()

def moveIn(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.x -= dist
    robot.set_pose(currentState)
    robot.close()

def moveOut(dist):
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.x += dist
    robot.set_pose(currentState)
    robot.close()

def get_sensor_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("160.69.69.110",8190))

    msg  = s.recv(1028)
    s.close()

    print(msg)



#def capture():




if __name__ == '__main__':
    
    
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_EXPOSURE, -10) 
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
            _thread.start_new_thread( moveLeft, (0.03,) )
            #moveLeft(0.03)
            print("moved Left")
            get_sensor_data()
        elif(chr(keyInput & 255) == "d"):
            _thread.start_new_thread( moveRight, (0.03,) )            
            print("moved right")
            get_sensor_data()
        elif (chr(keyInput & 255)== "s"):
            
            _thread.start_new_thread( moveDown, (0.03,) )
            print("moved down")
            get_sensor_data()
        elif(chr(keyInput & 255)=="w"):
            
            _thread.start_new_thread( moveUp, (0.03,) )
            print("moved up")
            get_sensor_data()
        elif(chr(keyInput & 255)=="i"):
            
            _thread.start_new_thread( moveIn, (0.03,) )
            print("moved In")
            get_sensor_data()
        elif(chr(keyInput & 255)=="o"):
            
            _thread.start_new_thread( moveOut, (0.03,) )
            print("moved Out")
            get_sensor_data()
            
        elif(chr(keyInput & 255) == "x"):
            get_sensor_data()

        elif(chr(keyInput & 255) == "c"):
            img_name = "img{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            img_counter += 1 
            print("picture taken")



            #code to take a snap shot and save in a specifivc folder 
