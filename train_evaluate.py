import os
import argparse
import numpy as np
from utils import save_results, init_classifier

import tensorflow as tf

def main(args):
	with open('data/train_splits_undersampled.npz', 'rb') as load_file:
		npzfile = np.load(load_file)    
		
		X_train = npzfile['X_train']
		X_val = npzfile['X_val']
		X_test = npzfile['X_test']
		
		y_train = npzfile['y_train']
		y_val = npzfile['y_val']
		y_test = npzfile['y_test']

	outputs_dir = 'results/{}/'.format(args.model)
	os.makedirs('results/', exist_ok=True)
	os.makedirs(outputs_dir, exist_ok=True)

	input_shape = X_train.shape[1:]
	nb_classes 	= 2
	verbose 	= True

	classifier = init_classifier(args.model, outputs_dir, input_shape, nb_classes, verbose)
		
	#classifier.fit(X_train, y_train, X_val, y_val)

	#y_pred = classifier.predict(X_test)

	#save_results(args.model, np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--model', type=str)

	# TODO: Add all model params as arguements

	# Transformer
	'''
	parser.add_argument('-emb_dim', type=int, default=100) # Embedding size for each token
	parser.add_argument('-num_heads', type=int, default=2) # Number of attention heads
	parser.add_argument('-ff_dim', type=int, default=32) # Hidden layer size in feed forward network inside transformer
	'''

	args = parser.parse_args()
	
	main(args)