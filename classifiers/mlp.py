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

class Classifier_MLP(base_Model):
	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
		super(Classifier_MLP, self).__init__()

		self.output_directory = output_directory

		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			
			if verbose==True:
				self.model.summary()
		
		self.verbose = verbose
		
		file_path = self.output_directory+'best_model.hdf5'
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
		self.callbacks.append(model_checkpoint)

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		# flatten/reshape because when multivariate all should be on the same axis 
		input_layer_flattened = keras.layers.Flatten()(input_layer)
		
		layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
		layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

		layer_2 = keras.layers.Dropout(0.2)(layer_1)
		layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

		layer_3 = keras.layers.Dropout(0.2)(layer_2)
		layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

		output_layer = keras.layers.Dropout(0.3)(layer_3)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		optimizer = keras.optimizers.Adam(lr = 0.01)
		model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=self.metrics)
		tf.keras.utils.plot_model(model, to_file='models/mlp_plot.png', show_shapes=True, show_layer_names=True)

		session.close()

		return model
