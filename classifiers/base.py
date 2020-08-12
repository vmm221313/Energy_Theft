import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from utils import matthews_correlation
from classifiers.custom_layers import TokenAndPositionEmbedding, TransformerBlock

class base_Model():
	def __init__(self):
		super(base_Model, self).__init__()

		# Callbacks 
		#reduce_lr 				= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
		#model_checkpoint 		= keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
		earlystop 				= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
		self.callbacks 			= [earlystop]

		# Metrics 
		AUC 					= tf.keras.metrics.AUC()
		Accuracy 				= tf.keras.metrics.BinaryAccuracy()
		TruePositives 			= tf.keras.metrics.TruePositives()
		TrueNegatives			= tf.keras.metrics.TrueNegatives()
		FalsePositives			= tf.keras.metrics.FalsePositives()
		FalseNegatives			= tf.keras.metrics.FalseNegatives()
		Precision				= tf.keras.metrics.Precision()
		Recall					= tf.keras.metrics.Recall()
		MCC  					= matthews_correlation#tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
		self.metrics 			= [AUC, Accuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Precision, Recall, MCC]

	def fit(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=100):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		
		# Fit Model
		hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
		
		# Save last model
		#self.model.save(self.output_directory+'last_model.hdf5')

		# Load best model
		#model = keras.models.load_model(self.output_directory+'best_model.hdf5')
		self.model.load_weights(self.output_directory+'best_model.hdf5')
		#load_status.assert_consumed()

		#y_pred = self.model.predict(x_val) 
		#y_pred = np.argmax(y_pred , axis=1) # convert the predicted from binary to integer

		keras.backend.clear_session()

		return hist

	def predict(self, x_test, model_path=None):
		
		if model_path == None:
			model_path = self.output_directory+'best_model.hdf5'

		#model  = keras.models.load_model(model_path)
		self.model.load_weights(self.output_directory+'best_model.hdf5')
		y_pred = self.model.predict(x_test)

		return y_pred