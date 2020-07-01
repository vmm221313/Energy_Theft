import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from classifiers.base import base_Model
#from utils.utils import save_logs
#from utils.utils import calculate_metrics

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
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
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

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

		return model
