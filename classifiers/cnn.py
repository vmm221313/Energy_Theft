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

class Classifier_CNN(base_Model):
	def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
		super(Classifier_CNN, self).__init__()
		
		self.output_directory = output_directory

		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			
			if verbose == True:
				self.model.summary()
		
		self.verbose = verbose

		file_path = self.output_directory+'best_model.hdf5'
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
		self.callbacks.append(model_checkpoint)

	def build_model(self, input_shape, nb_classes):
		padding = 'valid'
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
		conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

		conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
		conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

		flatten_layer = keras.layers.Flatten()(conv2)
		output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid')(flatten_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		
		optimizer = keras.optimizers.Adam(lr = 0.01)
		model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=self.metrics)
		tf.keras.utils.plot_model(model, to_file='models/cnn_plot.png', show_shapes=True, show_layer_names=True)

		session.close()

		return model

