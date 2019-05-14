from __future__ import print_function

import argparse
import time
import numpy as np
import cv2 as cv
import random
import math
#from scipy.misc import imread

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


#docker run --name rivetserving -e MODEL_NAME="rivetQmodel" -p 8501:8501 --mount type=bind,source=C:/Users/gulat/Desktop/thesis/gitThesis/ServingModels/rivetModel,target=/models/rivetQmodel -t tensorflow/serving 
#C:\Users\gulat\Desktop\thesis\gitThesis\testScripts>python servingClient.py --image C:\Users\gulat\Desktop\img17.png --model rivetQmodel --host "172.0.0.2" --port 8500 --signature_name serving_default

#working

#docker run --rm -p 8500:8500 -v C:/Users/gulat/Desktop/thesis/gitThesis/ServingModels/rivetModel/rivetQmodel:/models/rivetQmodel -e MODEL_NAME="rivetQmodel" -e MODEL_PATH="/models/rivetQmodel" --name rivetserving tensorflow/serving
#C:\Users\gulat\Desktop\thesis\gitThesis\testScripts> python servingClient.py --image C:\Users\gulat\Desktop\zeroDegEvalGood\img0.png --model rivetQmodel --host "127.0.0.1" --port 8500 --signature_name serving_default

def run(host, port, image, model, signature_name):

    allGood = ["All is well","HAKUNA MATATA","AAll's GUT","Next Please","Yeah this will work","Damn Lookin Fine !!!!"]
    allBad =["Sir Please step aside for further inspection !!", "Aaah you need some working","Serously !! you thought riveting is easy !!","DENIED", "You need some workin man !!" ]
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Read an image
    data = cv.imread(image)
    print("hello")

    height, width ,depth= data.shape
    data = data[(math.ceil(height/2)-50):(math.ceil(height/2)+50), (math.ceil(width/2)-50):(math.ceil(width/2)+50)]
    height, width ,depth= data.shape
    #data =cv.resize(data,(math.ceil(width), math.ceil(height)))

    data = cv.blur(data,(5,5))


    data = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    data = cv.Canny(data,50,90)

    cv.imshow("test",data)    
    cv.waitKey(0)
    cv.destroyAllWindows()

    data =cv.resize(data,(42,42))

    


    data = data.astype(np.float32)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs["x"].CopyFrom(make_tensor_proto(data, shape=[1, 36, 36, 1]))

    result = stub.Predict(request, 10)


    badPrediction = result.outputs["probabilities"].float_val[0]
    goodPrediction = result.outputs["probabilities"].float_val[1]
    end = time.time()
    time_diff = end - start
    bad = "B " + str(allBad[random.randint(0,len(allBad)-1)])
    good = "G " + str(allGood[random.randint(0,len(allGood)-1)])
    if(badPrediction>goodPrediction):
        #print("B "+ allBad[random.randint(0,len(allBad))])
        return bad
    elif(goodPrediction>badPrediction):
        #print("G "+ allGood[random.randint(0,len(allGood))])
        return good        
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':


    for i in range(0,18,1):

        print("image_"+str(i) + " :"+ run("127.0.0.1","8500","C:/Users/gulat/Desktop/nnnnnnnnn/img"+str(i)+".png","rivetQmodel","serving_default"))
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='0.0.0.0', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',default='serving_default', type=str)

    args = parser.parse_args()
    
    run(args.host, args.port, args.image, args.model,args.signature_name)
    '''