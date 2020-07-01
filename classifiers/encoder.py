import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from classifiers.base import base_Model

tf.random.set_seed(42)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow_addons as tfa

class Classifier_ENCODER(base_Model):
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        super(Classifier_ENCODER, self).__init__()

        self.output_directory = output_directory
        
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            
            if verbose == True:
                self.model.summary()
        
        self.verbose = verbose

        file_path = self.output_directory+'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
        self.callbacks.append(model_checkpoint)

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        optimizer = keras.optimizers.Adam(lr = 0.01)

        model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

        session.close()

        return model
