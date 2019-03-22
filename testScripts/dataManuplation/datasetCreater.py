from random import shuffle
import glob
import sys
import cv2
import numpy as np
#import skimage.io as io
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    # read an image and resize to (36,36)
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(36,36))
    return img


def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 100:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])
        
        label = labels[i]
        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

if __name__ == "__main__":

    allImageDirectory = 'C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/*/*.png'
    # list in address of all images
    addrs =[]
    for i in range(0,10549,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/bad/img"+str(i)+".png")
    
    for i in range(0,10409,1):
        addrs.append("C:/Users/gulat/Desktop/thesis/gitThesis/images/organisedImages/good/img"+str(i)+".png")

    addrs1 = glob.glob(allImageDirectory)
    labels=[]
    for addr in addrs:
        if("bad" in addr):
            labels.append(0)
        elif("good" in addr):
            labels.append(1)



    
    #print(labels)
    #labels = [0 if 'bad' in addr else for addr in addrs]  # 0 = Cat, 1 = Dog
    #print(str(len(addrs)))

    #print(str(len(addrs1)))
    
    
    # to shuffle data
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    #print(labels)
    for k in range(20,48,1):
        print(str(labels[len(labels)-k]) + " #### "+ str(addrs[len(addrs)-k]) )


    '''
    # Divide the data into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]

    createDataRecord('train.tfrecords', train_addrs, train_labels)
    createDataRecord('val.tfrecords', val_addrs, val_labels)
    createDataRecord('test.tfrecords', test_addrs, test_labels)
    '''