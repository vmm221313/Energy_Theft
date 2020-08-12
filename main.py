import os
import argparse
import numpy as np

import tensorflow as tf

from utils import save_results, init_classifier, load_data

def main(args):
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.train_frac, args.val_frac, args.test_frac, type=args.imbalance_type)

	outputs_dir = 'results/{}/'.format(args.model)
	os.makedirs('results/', exist_ok=True)
	os.makedirs(outputs_dir, exist_ok=True)

	input_shape = X_train.shape[1:]
	nb_classes 	= 2
	verbose 	= True

	classifier = init_classifier(args.model, outputs_dir, input_shape, nb_classes, verbose)
		
	# TODO -> use class weights

	hist = classifier.fit(X_train, y_train, X_val, y_val, batch_size=64)
	#print(hist.history)

	y_pred = classifier.predict(X_test)

	save_results(args.model, y_test, y_pred, hist.history, outputs_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Classifier to be used
	parser.add_argument('--model', type=str, choices=['cnn', 'encoder', 'fcn', 'mlp', 'resnet', 'transformer', 'lstmfcn', 'alstmfcn'])
	
	# Imbalance Technique to be used
	parser.add_argument('--imbalance_type', type=str, default='standard')
	
	# Train-Validation-Testing Distribution 
	parser.add_argument('--train_frac', type=float, default=0.6)	
	parser.add_argument('--val_frac', type=float, default=0.2)	
	parser.add_argument('--test_frac', type=float, default=0.2)	

	# TODO: Add all model params as arguements

	# Transformer Params
	parser.add_argument('--emb_dim', type=int, default=100) # Embedding size for each token
	parser.add_argument('--num_heads', type=int, default=2) # Number of attention heads
	parser.add_argument('--ff_dim', type=int, default=32) # Hidden layer size in feed forward network inside transformer

	args = parser.parse_args()
	
	main(args)