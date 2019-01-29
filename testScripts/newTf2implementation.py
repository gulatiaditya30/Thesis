
#GOOD --> 1
#BAD --> 0
#ideal edge detection range  = 50-120


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
import cv2 as cv

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer 
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 36, 36, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 9 * 9 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dense2 = tf.layers.dense(inputs = dense, units = 100, activation =tf.nn.relu)

  dense3  = tf.layers.dense(inputs = dense2 , units  = 60 ,activation = tf.nn.relu)


  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(inputs=dense3, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 60]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



#====================================================my input function ===============================================

def image_Input_func(trainData,labels,noOfEpoch = 1,batchSize =1 ,shuffle = True,repeatCount = 1): #input function which converts raw data into dataset and sets up the batch,epoch ... parameters and return a tuple containing feature and label values

    '''def _parse_function(trainD, label): #not required as im returning a dict in the end wrt to batch

        d = dict(zip(['inputlevelImage'], trainD)),label

        return d
'''

    dataset = tf.data.Dataset.from_tensor_slices((trainData,labels))
    

    #dataset = dataset.map(_parse_function)

    dataset = dataset.repeat(noOfEpoch)
    dataset = dataset.batch(batchSize)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return {'x':batch_features},batch_labels
#==============================================================================================================
def validityCheck(plateNo,rivetNo):
    with open('C:/Users/gulat/Desktop/thesis/gitThesis/images/labeledPlate.csv')as labbeledFile:
        fileRead = csv.reader(labbeledFile,delimiter = ',')
        x = list(fileRead)
        if(x[rivetNo][plateNo+1] == 'NA'):
            return 'NA'
        else:
            return str(x[rivetNo][plateNo+1])

#================================================== creating tf record ==================================================

def dataAddressGen():
    addrs = []
    for i in range(0,11087,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/trainBad/img"+str(i)+".png")

    for i in range(0,11087,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/trainGood/img"+str(i)+".png")

    labels= []
    for addr in addrs:
        if("trainBad" in addr):
            labels.append(0)
        elif("trainGood" in addr):
            labels.append(1)
    
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
    return addrs,labels


#=========================================================================
def evalDataAddressGen():
    addrs = []
    for i in range(0,50,1):#(0,10549,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/zeroDegEvalBad/img"+str(i)+".png")

    for i in range(0,50,1):#(0,10409,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/zeroDegEvalGood/img"+str(i)+".png")

    labels= []
    for addr in addrs:
        if("zeroDegEvalBad" in addr):
            labels.append(0)
        elif("ZeroDegEvalGood" in addr):
            labels.append(1)
    
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
    return addrs,labels


    




#===================================================== main function ====================================================

def main1():

  # Load training and eval data
  train_data = []
  eval_data = []

  train_labels = []
  eval_labels =[]

  
  allAddress,allLabels = dataAddressGen()
  evalAddress,evalLabels = evalDataAddressGen()

  trainAdd = allAddress
  train_labels = allLabels
  train_labels = list(train_labels)

  evalAdd = evalAddress
  eval_labels =evalLabels
  eval_labels = list(eval_labels)
  '''
  trainAdd = allAddress[0:int(0.80*len(allAddress))]
  train_labels = allLabels[0:int(0.80*len(allLabels))]
  train_labels = list(train_labels)

  print("how u like me now !! "+ str(type(train_data)))
  

  evalAdd = allAddress[int(0.80*len(allAddress)):]
  eval_labels = allLabels[int(0.80*len(allLabels)):]
  eval_labels = list(eval_labels)
'''
  '''
  for i in range(1000,4000,1):
      img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/bad/img"+str(i)+".png")
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      img = cv.resize(img,(36,36))
      train_data.append(img)
      train_labels.append(0)
      
  for i in range(0,1001,1):
      img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/bad/img"+str(i)+".png")
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      img = cv.resize(img,(36,36))
      eval_data.append(img)
      eval_labels.append(0)

  '''
  for tradd in trainAdd:
      img = cv.imread(tradd)
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      img = cv.resize(img,(36,36))
      train_data.append(img)
 
  for evadd in evalAdd:
      img = cv.imread(evadd)
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      img = cv.resize(img,(36,36))
      eval_data.append(img)
   

  '''
  for z in range (0,181,30):
    for x in range(0,9,1):         #plates
          for y in range(0,280,1):
              if(x == 0):
                if(validityCheck(x,y) != 'NA' ):
                    eval_labels.append(int(validityCheck(x,y)))
                    img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/postProcessedImage/degree"+str(z)+"/plate"+str(x)+"/img"+str(y)+".png")
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    img = cv.resize(img,(40,40))
                    eval_data.append(img)
              else:
                if(validityCheck(x,y) != 'NA' ):
                    train_labels.append(int(validityCheck(x,y)))
                    img = cv.imread("C:/Users/gulat/Desktop/thesis/gitThesis/images/postProcessedImage/degree"+str(z)+"/plate"+str(x)+"/img"+str(y)+".png")
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    img = cv.resize(img,(40,40))
                    train_data.append(img)
                    '''
  train_data = np.array(train_data,dtype='float32')
  train_data = np.reshape(train_data,[train_data.shape[0],train_data.shape[1]*train_data.shape[2]])

  eval_data = np.array(eval_data,dtype='float32')
  eval_data = np.reshape(eval_data,[eval_data.shape[0],eval_data.shape[1]*eval_data.shape[2]])
 
 




  print ("444444444444444444444444444444444444444444444444444444444444444444444444444444444444444")

  print(eval_data[0])

  print(str(len(eval_data))+"  ==  "+str(len(eval_labels)) +" == "+ str(len(evalAdd)))
  print(str(len(train_data))+"  ==  "+str(len(train_labels))+ " == "+ str(len(trainAdd)))

  
  k=0
  print(str(eval_labels[k]) +" == "+ str(evalAdd[k]))
  print(str(train_labels[k])+ " == "+ str(trainAdd[k]))
  

  

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
       #  tensors=tensors_to_log, every_n_iter=50)


  mnist_classifier.train(input_fn = lambda : image_Input_func(trainData = train_data,labels =train_labels,noOfEpoch = 1,batchSize = 10 ,shuffle = True,repeatCount = 1))
  evaluate_results = mnist_classifier.evaluate(input_fn = lambda : image_Input_func(trainData = eval_data, labels = eval_labels, batchSize = 10, shuffle = False , repeatCount = 1))
  print(evaluate_results)

  #print(evaluation_result)
  #for key in evaluation_result:
#      print("  {} ,  was {}".format(key, evaluate_result[key]))


print("hello")
main1()
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Done ")




#if __name__ == "__main__":

 # tf.app.run()
