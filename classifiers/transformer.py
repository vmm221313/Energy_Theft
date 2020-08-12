import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from classifiers.base import base_Model
from classifiers.custom_layers import TokenAndPositionEmbedding, TransformerBlock

tf.random.set_seed(42)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class Classifier_TRANSFORMER(base_Model):
	def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
		super(Classifier_TRANSFORMER, self).__init__()

		self.output_directory = output_directory
		
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)

			if(verbose==True):
				self.model.summary()

		self.verbose = verbose
		
		file_path = self.output_directory+'best_model.hdf5'
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
		self.callbacks.append(model_checkpoint)
	
	def build_model(self, input_shape, nb_classes):
		emb_dim 	= 512
		num_heads	= 2
		ff_dim 		= 128
		transformer_block = TransformerBlock(emb_dim, num_heads, ff_dim)

		input_layer = keras.layers.Input(input_shape)
		x = transformer_block(input_layer)
		x = keras.layers.GlobalAveragePooling1D()(x)
		x = keras.layers.Dropout(0.1)(x)
		x = keras.layers.Dense(20, activation="relu")(x)
		x = keras.layers.Dropout(0.1)(x)
		output_layer = keras.layers.Dense(nb_classes, activation="softmax")(x)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		
		optimizer = keras.optimizers.Adam(lr = 1)
		model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=self.metrics)
		tf.keras.utils.plot_model(model, to_file='models/transformer_plot.png', show_shapes=True, show_layer_names=True)

		session.close()
		return model 
