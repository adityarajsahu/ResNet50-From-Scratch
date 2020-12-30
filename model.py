#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Add, ZeroPadding2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

"""
x and x_skip are two matrices, so they can be added
only when they have the same shape. If after 
convolution and batch normalization operations are
done, the shape of x and x_skip are same, then these
two layers are added directly.
"""

def ResNet_IdentityBlock(x, filters):
    x_skip = x
    f1, f2 = filters

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x

"""
If after convolution and batch normalization operations
are done, the shape of x and x_skip are not same, then
we pass the x_skip through a convolution-batch normalization
layer such that the shape of x and x_skip becomes same.
Then, the x and x_skip are added.
"""

def ResNet_ConvBlock(x, s, filters):
    x_skip = x
    f1, f2 = filters

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    x_skip = Conv2D(filters=f2, kernel_size=(1, 1), strides=(s, s), padding='valid')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x

"""
The Convolution blocks and Identity blocks are 
arranged in the way, it is described in the 
Residual Network research paper.
"""

def ResNet50(input_size=(224, 224, 3), output=5):
    # Input image size = (224, 224, 3)
    # Output size = (5, 1)
    input = Input(input_size)
    # Padding of zeros is applied around the image
    x = ZeroPadding2D(padding=(3, 3))(input)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = ResNet_ConvBlock(x, s=1, filters=(64, 256))
    x = ResNet_IdentityBlock(x, filters=(64, 256))
    x = ResNet_IdentityBlock(x, filters=(64, 256))

    x = ResNet_ConvBlock(x, s=2, filters=(128, 512))
    x = ResNet_IdentityBlock(x, filters=(128, 512))
    x = ResNet_IdentityBlock(x, filters=(128, 512))
    x = ResNet_IdentityBlock(x, filters=(128, 512))

    x = ResNet_ConvBlock(x, s=2, filters=(256, 1024))
    x = ResNet_IdentityBlock(x, filters=(256, 1024))
    x = ResNet_IdentityBlock(x, filters=(256, 1024))
    x = ResNet_IdentityBlock(x, filters=(256, 1024))
    x = ResNet_IdentityBlock(x, filters=(256, 1024))
    x = ResNet_IdentityBlock(x, filters=(256, 1024))

    x = ResNet_ConvBlock(x, s=2, filters=(512, 2048))
    x = ResNet_IdentityBlock(x, filters=(512, 2048))
    x = ResNet_IdentityBlock(x, filters=(512, 2048))

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(output, activation='softmax')(x)

    # model is defined
    model = Model(inputs=input, outputs=x) 
    
    adam = Adam(lr=1e-4)
    # Optimizer, Loss function & metrics are defined for the model
    model.compile(optimizer=adam, loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

    return model
