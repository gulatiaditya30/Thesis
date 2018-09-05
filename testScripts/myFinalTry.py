from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt


import cv2 as cv
import math


import numpy as np
import tensorflow as tf

def conv_nn_model_fn(inputImg,labels,mode):
    #setting up input layer vector
    inputLayer = tf.reshape(inputImg["x"],[-1,500,500,1])

    #setting up first convolution layer
    #in  = [500,500,1]
    #out = [500,500,32]
    convLayer1 = tf.layers.conv2d(inputs = inputLayer,filter =32 ,kernel_size = [5,5], padding = "same", activation = tf.nn.relu )

    #setting up pooling layer 1
    #in = [500,500,32]
    #out = [250,250,32]
    poolingLayer1 = tf.layer.max_pooling2d(inputs =convLayer1, pool_size = [2,2], strides = 2)

    #setting up seconf convolution layer2
    #in = [250,250,32]
    #out = [250,250,64] # the output has 64 channels only and not (32 * 64) because the convolution filter runs its 5,5 filter on all its 32 channels and give one output only
    convLayer2 = tf.layers.conv2d(inputs = poolingLayer1 , filter =64,kernel_size = [5,5], padding ="same", activation =tf.nn.relu)

    #setting up pooling layer for convLayer2
    #in = [250,250,64]
    #out = [125,125,64]
    poolingLayer2 = tf.layer.max_pooling2d(inputs = convLayer2, pool_size = [2,2], strides = 2)

    #flating up the poolimage for  dense layer
    poolLayerFlat = tf.reshape(inputs = poolingLayer2, [-1,125*125*64])

    #setting up a dense layer
    denseLayer1 = tf.layers.dense(inputs = poolLayerFlat,units = 1024,activation = tf.nn.relu)

    #seeting up hidden dense layer 2
    denseLayer2 = tf.layer.dense(inputs = denseLayer1,units =100,activation = tf.nn.relu)

    #adding a drop out layer to make it more non linear ( avoiding overfitting)
    dropout = tf.layers.dropout(inputs = denseLayer2, rate= 0.4,training =mode == tf.estimator.ModeKeys.TRAIN)

    #final output layer
    logits = tf.layers.dense(inputs = dropout,units =10)

    # the model is set ###########################################################################


    #setting up prediction dictionary for evaluation

    predictions = {
        "rivetQualityClass" : tf.argmax(input = logits, axis =1),
        "probabilities" : tf.nnsoftmax(logits,name = "softmax_tensor")
    }

    #=================================== different mode details ======================================
    #loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    # configuring predict mode
    if mode == tf.estimator.Modekeys.PREDICT:
        return tf.estimator.EsmatorSpec(mode = mode , predictions = predictions)

    # configuring Train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
        train_op = optimizer.minimize(loss = loss,global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss , train_op = train_op)

    #configure for evaluation mode
    if mode ==tf.estimator.ModeKEYS.EVAL:{
            "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#=========================================== data pre processing ================================

def dataPreProcessing():
    data = []
    for i in range (1,5):
        img = cv.imread('C:/Users/gulat/Desktop/thesis/testScripts/rivetEdge'+str(i)+'.jpg',-1)
        height, width ,depth= img.shape
        img = img[(math.ceil(height/2)-500):(math.ceil(height/2)+500), (math.ceil(width/2)-500):(math.ceil(width/2)+500)]

        height, width ,depth= img.shape
        img =cv.resize(img,(math.ceil(width), math.ceil(height)))

        img = cv.blur(img,(5,5))

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        height, width = gray.shape

        edges = cv.Canny(gray,100,150)

        #img = cv.resize(img,( math.ceil(640), math.ceil(6)))

        flatEdge = edges.ravel()
        data.append(flatEdge)
        print (len(data))

        return img


#========================================= setting up dataset for tf estimator ================================

def getImage(fileName,label):

    image_reader = tf.read_file(filename)
    img = tf.image.decode_jpeg(image_reader)
    img.set_shape([150,150,1])

    #still need to do add image preprocessing

    return img,label


def image_Input_func(fileNames,labels,noOfEpoch,batchSize =1 ,shuffle = True,repeateCount = 1,): #input function which converts raw data into dataset and sets up the batch,epoch ... parameters and return a tuple containing feature and label values
    fileNames = tf.constant(fileNames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((fileNames,labels))
    dataset.map(getImage())  #map functions apply the passed function to all the slices

    dataset = dataset.repeat(noOfEpoch)
    dataset = dataset.batch(batchSize)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels









#========================================== main function implementation ======================================


def main(unused_argv):
    # set up training data
