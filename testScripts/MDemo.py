import urx 
import msvcrt
import cv2 as cv
import socket
import _thread
import csv
import math
import time

import numpy as np

import random
import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


moveDist = 0.01
img_counter = 0
plateCounter = 0
exposureV = -2
#ideal edge detection range  = 50-120 ?? working 50-100
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



def moveRobot():

    global msgCollectFlag
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    #currentState.pos.y -= 0.075
    #robot.set_pose(currentState,vel = 5)
    currentState.pos.y += 0.02
    msgCollectFlag = True
    robot.set_pose(currentState,vel = 0.005,acc =2 )
    msgCollectFlag =False

def moveRobotBack():
    robot  = urx.Robot("160.69.69.101")
    currentState = robot.get_pose()
    currentState.pos.y -= 0.02
    robot.set_pose(currentState,vel = 0.005,acc =2 )

def imgPreProcessing(imgAdd):
    data = []
    img = imgAdd
    height, width ,depth= img.shape
    img = img[(math.ceil(height/2)-50):(math.ceil(height/2)+50), (math.ceil(width/2)-50):(math.ceil(width/2)+50)]
    height, width ,depth= img.shape
    #img =cv.resize(img,(math.ceil(width), math.ceil(height)))

    img = cv.blur(img,(5,5))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    

    edges = cv.Canny(gray,50,100)

    #cv.imshow('image',edges)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    

    #===============================================================================================

    allGood = ["All is well","HAKUNA MATATA","AAll's GUT","Next Please","Yeah this will work"]#"Damn Lookin Fine !!!!"]
    allBad =["Sir Please step aside for further inspection !!", "Aaaah you need some working","Serously !! you thought riveting is easy !!","DENIED"]
    channel = grpc.insecure_channel("127.0.0.1:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Read an image
    data = edges
    print("hello")
    data =cv.resize(data,(36,36))

    data = data.astype(np.float32)
    print(data.shape)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "rivetQmodel"
    request.model_spec.signature_name = "serving_default"
    request.inputs["x"].CopyFrom(make_tensor_proto(data, shape=[1, 36, 36, 1]))

    result = stub.Predict(request, 10)

    
    badPrediction = result.outputs["probabilities"].float_val[0]
    goodPrediction = result.outputs["probabilities"].float_val[1]
    end = time.time()
    time_diff = end - start
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@***********************************@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Good Pre:"+ str(goodPrediction))
    print("bad Pre: "+str(badPrediction))
    if(badPrediction>goodPrediction):
        print("BAD : "+ allBad[int(random.randint(0,len(allBad)-1))])
    elif(goodPrediction>badPrediction):
        print("GOOD : "+ allGood[int(random.randint(0,len(allGood)-1))])
    print('time elapased: {}'.format(time_diff))
    





if __name__ == '__main__':
    
    global msgCollectFlag 
    msgCollectFlag= False
    cam = cv.VideoCapture(1)
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
            _thread.start_new_thread(moveLeft, (.02,) )
            #moveLeft(0.03)
            print("moved Left")
            
            #get_sensor_data()
           
        elif(chr(keyInput & 255) == "d"):
            _thread.start_new_thread(moveRight, (.02,) )            
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
        
        elif(chr(keyInput & 255) == "c"):
            img_name = "C:/Users/gulat/Desktop/nnnnnnnnn/img"+str(img_counter)+".png"
            #cv.imwrite(img_name, frame)
            imgPreProcessing(frame)

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
