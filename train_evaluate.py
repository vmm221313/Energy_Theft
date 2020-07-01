import os
import argparse
import numpy as np
from utils import save_results
#from classifiers.fcn import Classifier_FCN
#from classifiers.resnet import Classifier_RESNET
'''


model_name = 'resnet'

outputs_dir = 'results/{}/'.format(model_name)



classifier = Classifier_RESNET(outputs_dir=outputs_dir, input_shape=X_train.shape[1:], nb_classes=2, verbose=True)

'''
#resnet = Classifier_RESNET(outputs_dir=outputs_dir, input_shape=X_train.shape[1:], nb_classes=2, verbose=True)
#resnet.fit(X_train, y_train, X_val, y_val)

def init_classifier(classifier_name, outputs_dir, input_shape, nb_classes=2, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(outputs_dir, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(outputs_dir, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(outputs_dir, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(outputs_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(outputs_dir, input_shape, nb_classes, verbose)

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

	y_pred = classifier.predict(X_test)

	save_results(args.model, np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--model', type=str)

	args = parser.parse_args()
	
	main(args)