import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class base_Model():
	def __init__(self):
		super(base_Model, self).__init__()

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
		#model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
		earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
		
		self.callbacks = [earlystop]

	def fit(self, x_train, y_train, x_val, y_val):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 64
		nb_epochs = 100

		#mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		#save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, model_path=None):
		
		if model_path == None:
			model_path = self.output_directory+'best_model.hdf5'

		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)

		return y_pred