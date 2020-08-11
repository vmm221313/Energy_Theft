import numpy as np 
import pandas as po

import tensorflow as tf

def get_standard_data(train_frac, val_frac, test_frac):
	print('Loading Standard Data with train_frac = {}, val_frac = {} and test_frac = {}'.format(train_frac, val_frac, test_frac))
	
	df = po.read_csv('data/processed/imputation/edtwbi.csv').sample(frac=1, random_state=42).reset_index(drop=True)
	
	X = df.drop(['num_zeros', 'FLAG'], axis=1)
	y = df['FLAG']

	print('Value counts for each class - ')
	print(y.value_counts())

	X_train = X[:int(train_frac*len(X))]
	y_train = y[:int(train_frac*len(X))]

	X_val   = X[int(train_frac*len(X)):int((train_frac+val_frac)*len(X))]
	y_val   = y[int(train_frac*len(X)):int((train_frac+val_frac)*len(X))]

	X_test  = X[int((train_frac+val_frac)*len(X)):int((train_frac+val_frac+test_frac)*len(X))]
	y_test  = y[int((train_frac+val_frac)*len(X)):int((train_frac+val_frac+test_frac)*len(X))]

	X_train = X_train.to_numpy()[:, :, np.newaxis]
	X_val   = X_val.to_numpy()[:, :, np.newaxis]
	X_test  = X_test.to_numpy()[:, :, np.newaxis]

	y_train = tf.keras.utils.to_categorical(y_train.to_numpy(), num_classes=2)
	y_val   = tf.keras.utils.to_categorical(y_val.to_numpy(), num_classes=2)
	y_test  = tf.keras.utils.to_categorical(y_test.to_numpy(), num_classes=2)

	return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_undersampled_data(train_frac, val_frac, test_frac):
	print('Loading Undersampled Data with train_frac = {}, val_frac = {} and test_frac = {}'.format(train_frac, val_frac, test_frac))
	
	df = po.read_csv('data/processed/imputation/edtwbi.csv').sample(frac=1, random_state=42).reset_index(drop=True)
	
	df_0 = df[df['FLAG'] == 0]
	df_1 = df[df['FLAG'] == 1]
	df_0 = df_0.sort_values('num_zeros')[:len(df_1)]

	df_undersampled = po.concat([df_1, df_0], axis=0, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

	print('Value counts for each class - ')
	print(df_undersampled['FLAG'].value_counts())

	X_undersampled 			= df_undersampled.drop(['num_zeros', 'FLAG'], axis=1)
	y_undersampled 			= df_undersampled['FLAG']

	X_undersampled_train 	= X_undersampled[:int(train_frac*len(X_undersampled))]
	y_undersampled_train 	= y_undersampled[:int(train_frac*len(X_undersampled))]

	X_undersampled_val   	= X_undersampled[int(train_frac*len(X_undersampled)):int((train_frac+val_frac)*len(X_undersampled))]
	y_undersampled_val   	= y_undersampled[int(train_frac*len(X_undersampled)):int((train_frac+val_frac)*len(X_undersampled))]

	X_undersampled_test  	= X_undersampled[int((train_frac+val_frac)*len(X_undersampled)):int((train_frac+val_frac+test_frac)*len(X_undersampled))]
	y_undersampled_test  	= y_undersampled[int((train_frac+val_frac)*len(X_undersampled)):int((train_frac+val_frac)*len(X_undersampled))]

	X_undersampled_train 	= X_undersampled_train.to_numpy()[:, :, np.newaxis]
	X_undersampled_val   	= X_undersampled_val.to_numpy()[:, :, np.newaxis]
	X_undersampled_test  	= X_undersampled_test.to_numpy()[:, :, np.newaxis]

	y_undersampled_train 	= tf.keras.utils.to_categorical(y_undersampled_train.to_numpy(), num_classes=2)
	y_undersampled_val   	= tf.keras.utils.to_categorical(y_undersampled_val.to_numpy(), num_classes=2)
	y_undersampled_test  	= tf.keras.utils.to_categorical(y_undersampled_test.to_numpy(), num_classes=2)

	return (X_train, y_train), (X_val, y_val), (X_test, y_test)