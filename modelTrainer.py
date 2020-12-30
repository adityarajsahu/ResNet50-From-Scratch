#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
from dataProcessor import processor
from model import ResNet50

# Create empty lists for storing image file path
daisy_path = []
dandelion_path = []
roses_path = []
sunflowers_path = []
tulips_path = []

# Append the image file paths to the lists
for filename in os.listdir('flower_photos/daisy/'):
    if filename.endswith('.jpg'):
        daisy_path.append(os.path.join('flower_photos/daisy/', filename))

for filename in os.listdir('flower_photos/dandelion/'):
    if filename.endswith('.jpg'):
        dandelion_path.append(os.path.join('flower_photos/dandelion/', filename))
        
for filename in os.listdir('flower_photos/roses/'):
    if filename.endswith('.jpg'):
        roses_path.append(os.path.join('flower_photos/roses/', filename))

for filename in os.listdir('flower_photos/sunflowers/'):
    if filename.endswith('.jpg'):
        sunflowers_path.append(os.path.join('flower_photos/sunflowers/', filename))

for filename in os.listdir('flower_photos/tulips/'):
    if filename.endswith('.jpg'):
        tulips_path.append(os.path.join('flower_photos/tulips/', filename))

# Append all the image file paths to a single list
image_path = daisy_path + dandelion_path + roses_path + sunflowers_path + tulips_path

"""
Image labels:
[1,0,0,0,0] - Daisy
[0,1,0,0,0] - Dandelion
[0,0,1,0,0] - Roses
[0,0,0,1,0] - Sunflowers
[0,0,0,0,1] - Tulips
"""

# Create empty lists for storing labels
daisy_label = []
dandelion_label = []
roses_label = []
sunflowers_label = []
tulips_label = []

# Append the labels to the lists
for filename in os.listdir('flower_photos/daisy/'):
    if filename.endswith('.jpg'):
        daisy_label.append([1,0,0,0,0])
    
for filename in os.listdir('flower_photos/dandelion/'):
    if filename.endswith('.jpg'):
        dandelion_label.append([0,1,0,0,0])
    
for filename in os.listdir('flower_photos/roses/'):
    if filename.endswith('.jpg'):
        roses_label.append([0,0,1,0,0])

for filename in os.listdir('flower_photos/sunflowers/'):
    if filename.endswith('.jpg'):
        sunflowers_label.append([0,0,0,1,0])

for filename in os.listdir('flower_photos/tulips/'):
    if filename.endswith('.jpg'):
        tulips_label.append([0,0,0,0,1])

# Append all the labels to a single list
image_label = daisy_label + dandelion_label + roses_label + sunflowers_label + tulips_label
image_label = np.array(image_label)

"""
Pass image_path, image_label into processor function
for data preprocessing and divide the data into two 
dataset : Training data, Validation data
"""
TrainData, ValData = processor(image_path, image_label)

# Create ResNet-50 model for image classification
model = ResNet50()

# define the patience of our model.
earlystopping = EarlyStopping(patience=4)

# define the model checkpoint to store only the best weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)

# Train the model and validate it on validation dataset for saving weights
model.fit(TrainData, epochs=20, validation_data=ValData, callbacks=[earlystopping, checkpoint])
