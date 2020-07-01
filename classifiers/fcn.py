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

class Classifier_FCN(base_Model):
	def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
		super(Classifier_FCN, self).__init__()

		self.output_directory = output_directory
		
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)

			if(verbose==True):
				self.model.summary()

		self.verbose = verbose
		
		file_path = self.output_directory+'best_model.hdf5'
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
		self.callbacks.append(model_checkpoint)

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		optimizer = keras.optimizers.Adam(lr = 0.01)

		model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

		session.close()
		return model 
