#!/usr/bin/env python

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def image_normalize(image):
    # scale down all the pixel values to the range [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    return image

def image_modifier(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    #resize the image into [224, 224]
    image = tf.image.resize(image, [224,224])
    #pass the image through image_mormalize function
    image = image_normalize(image)
    return image, label

def processor(image_path, label_path):
    #shuffle the dataset
    image_path, label_path = shuffle(image_path, label_path, random_state=0)
    #pick the first 1500 images from the shuffle dataset for image classification
    imagePath = image_path[:1500]
    labelPath = label_path[:1500]
    #split the dataset into two parts: one for training and other for validation
    TrainImage, ValImage, TrainLabel, ValLabel = train_test_split(imagePath, labelPath, test_size=0.2)
    #convert slices of array into objects
    TrainLoader = tf.data.Dataset.from_tensor_slices((TrainImage, TrainLabel))
    ValLoader = tf.data.Dataset.from_tensor_slices((ValImage, ValLabel))
    #process the images using image_modifier function and shuffle the dataset.
    TrainLoader = (TrainLoader.shuffle(32).map(image_modifier, num_parallel_calls=4).batch(8))
    ValLoader = (ValLoader.shuffle(32).map(image_modifier, num_parallel_calls=4).batch(8))
    return TrainLoader, ValLoader