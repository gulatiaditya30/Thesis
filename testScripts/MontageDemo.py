import urx 
import msvcrt
import cv2 as cv
import socket
import _thread
import csv
import math
import time
import paho.mqtt.client as mqtt

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

    # ======================sending info=============================

    mqClient = mqtt.Client("c1")
    mqClient.connect("192.168.1.1", port=1883, keepalive=60, bind_address="")

    result =""

    if(badPrediction>goodPrediction):
        result = "BAD : "+ str(allBad[int(random.randint(0,len(allBad)-1))])
        
    elif(goodPrediction>badPrediction):
        result = "GOOD : "+ str(allGood[int(random.randint(0,len(allGood)-1))])
    
    print(result)
    mqClient.publish("demo/quality",result)



    print('time elapased: {}'.format(time_diff))
    





if __name__ == '__main__':
    
    global msgCollectFlag 
    msgCollectFlag= False
    cam = cv.VideoCapture(0)
    cam.set(15,-5) #range is from -1 to -13 from long exposure to short exposure
    #print(type(robot.get_pose()))

    while True:

        frame =  cv.imread("C:/Users/gulat/Desktop/nnnnnnnnn/img"+str(img_counter)+".png")
        cv.imshow("test",frame)    

        keyInput  =  cv.waitKey(1)
        
        if(chr(keyInput & 255) == "q"):
            exit()

        elif(chr(keyInput & 255) == "a"):
            if(img_counter != 0):
                img_counter = img_counter - 1

        elif(chr(keyInput & 255) == "d"):
            if(img_counter != 16):
                img_counter = img_counter+ 1 

        elif(chr(keyInput & 255) == "c"):
            
            imgPreProcessing(frame)




                





            #code to take a snap shot and save in a specifivc folder 
