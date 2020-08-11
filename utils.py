import os
import pickle
import numpy as np 
import pandas as po
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, auc, roc_curve, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

from preprocessing import get_standard_data, get_undersampled_data

def covariance(x, y):
	xy = x*y

	exp_x = np.mean(x)
	exp_y = np.mean(y)
	exp_xy = np.mean(xy)

	return exp_xy - exp_x*exp_y

def PearsonCorrelation(x, y):
	cov = covariance(x, y)

	std_x = np.std(x)
	std_y = np.std(y)

	return (cov/(std_x*std_y))

def load_data(train_frac, val_frac, test_frac, type='standard'):
	assert train_frac + val_frac + test_frac <= 1

	if type == 'standard':
		return get_standard_data(train_frac, val_frac, test_frac)

	elif type == 'undersampled':
		return get_undersampled_data(train_frac, val_frac, test_frac)

	'''
	with open('data/train_splits_{}.npz'.format(type), 'rb') as load_file:
		npzfile = np.load(load_file)    
		
		X_train = npzfile['X_train']
		X_val = npzfile['X_val']
		X_test = npzfile['X_test']
		
		y_train = npzfile['y_train']
		y_val = npzfile['y_val']
		y_test = npzfile['y_test']
	'''

class save_results():
	def __init__(self, model_name, y_true, y_pred, history, outputs_dir):
		super(save_results, self).__init__()

		self.outputs_dir	= outputs_dir + 'plots/'
		os.makedirs(self.outputs_dir, exist_ok=True)

		self.history 		= history

		self.load_metrics_df()
		self.save_predictions(y_true, y_pred)

		self.model_name 	= model_name
		self.y_true 		= np.argmax(y_true, axis = 1)
		self.y_pred 		= np.argmax(y_pred, axis = 1)
		self.y_prob_true 	= y_pred[:, 1]

		self.evalutate_metrics()
		self.save_metrics_df()
		self.make_plots()

	def save_predictions(self, y_true, y_pred): # for comparitive analysis later
		np.savez(self.outputs_dir+'predictions.npz', y_true=y_true, y_pred=y_pred)

		with open(self.outputs_dir+'history.pkl', 'wb') as f:
			pickle.dump(self.history, f)

	## Plots
	def make_plots(self):
		self.__plot_roc_curve__()
		self.__plot_precision_recall_curve__()
		self.__plot_train_val_loss__()
		self.__plot_val_accuracy__()
		self.__plot_precision_recall_with_epochs__()

	def __plot_precision_recall_with_epochs__(self):
		plt.plot(self.history['val_precision'], color='blue', label='Precision')
		plt.plot(self.history['val_recall'], color='orange', label='Recall')
		plt.xlabel('Epochs')
		plt.ylabel('Value')
		plt.legend()
		plt.savefig(self.outputs_dir+'val_precision_recall.png')
		plt.close()

	def __plot_val_accuracy__(self):
		plt.plot(self.history['binary_accuracy'], color='blue', label='Training Accuracy')
		plt.plot(self.history['val_binary_accuracy'], color='orange', label='Val Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(self.outputs_dir+'train_val_accuracy.png')
		plt.close()

	def __plot_train_val_loss__(self):
		plt.plot(self.history['loss'], color='blue', label='Training Loss')
		plt.plot(self.history['val_loss'], color='orange', label='Val Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(self.outputs_dir+'train_val_loss.png')
		plt.close()

	def __plot_roc_curve__(self):
		lr_fpr, lr_tpr, _ = roc_curve(self.y_true, self.y_prob_true)
		
		plt.plot(lr_fpr, lr_tpr, marker='.')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		plt.savefig(self.outputs_dir+'roc_curve.png')
		plt.close()

	def __plot_precision_recall_curve__(self):
		lr_precision, lr_recall, _ = precision_recall_curve(self.y_true, self.y_prob_true)
		lr_f1, lr_auc = f1_score(self.y_true, self.y_pred), auc(lr_recall, lr_precision)

		plt.plot(lr_recall, lr_precision, marker='.')
		plt.xlabel('Recall')
		plt.ylabel('Precision')

		plt.savefig(self.outputs_dir+'precision_recall_curve.png')
		plt.close()

	## Metrics
	def evalutate_metrics(self):
		metrics = {'Model' 		: self.model_name, 
					'AUC' 		: self.__area_under_the_curve__(), 
					'MAP' 		: self.__average_precision_score__(), 
					'Accuracy' 	: self.__accuracy_score__(),
					'F1' 		: self.__f1_score__(), 
					'Precision' : self.__precision__(),
					'Recall'	: self.__recall__()
					}

		self.results_df = self.results_df.append(metrics, ignore_index=True)

	def __area_under_the_curve__(self):
		return roc_auc_score(self.y_true, self.y_prob_true)

	def __average_precision_score__(self):
		return average_precision_score(self.y_true, self.y_pred)

	def __accuracy_score__(self):
		return accuracy_score(self.y_true, self.y_pred)

	def __f1_score__(self):
		return f1_score(self.y_true, self.y_pred, average='micro')

	def __precision__(self):
		return precision_score(self.y_true, self.y_pred, average='micro')

	def __recall__(self):
		return recall_score(self.y_true, self.y_pred, average='micro')

	def load_metrics_df(self):
		if not os.path.exists('results/results.csv'):
			self.results_df = po.DataFrame(columns = ['Model'])
		else:
			self.results_df = po.read_csv('results/results.csv')

	def save_metrics_df(self):
		self.results_df.to_csv('results/results.csv', index=False)

def init_classifier(classifier_name, outputs_dir, input_shape, nb_classes=2, verbose=False):
	if classifier_name == 'fcn':
		from classifiers import fcn
		return fcn.Classifier_FCN(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'alstmfcn':
		from classifiers import alstmfcn
		return alstmfcn.Classifier_ALSTMFCN(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'lstmfcn':
		from classifiers import lstmfcn
		return lstmfcn.Classifier_LSTMFCN(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'mlp':
		from classifiers import mlp
		return mlp.Classifier_MLP(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'resnet':
		from classifiers import resnet
		return resnet.Classifier_RESNET(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'mcnn':
		from classifiers import mcnn
		return mcnn.Classifier_MCNN(outputs_dir, verbose)

	elif classifier_name == 'tlenet':
		from classifiers import tlenet
		return tlenet.Classifier_TLENET(outputs_dir, verbose)

	elif classifier_name == 'twiesn':
		from classifiers import twiesn
		return twiesn.Classifier_TWIESN(outputs_dir, verbose)

	elif classifier_name == 'encoder':
		from classifiers import encoder
		return encoder.Classifier_ENCODER(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'mcdcnn':
		from classifiers import mcdcnn
		return mcdcnn.Classifier_MCDCNN(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'cnn':  # Time-CNN
		from classifiers import cnn
		return cnn.Classifier_CNN(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'inception':
		from classifiers import inception
		return inception.Classifier_INCEPTION(outputs_dir, input_shape, nb_classes, verbose)

	elif classifier_name == 'transformer':
		from classifiers import transformer
		return transformer.Classifier_TRANSFORMER(outputs_dir, input_shape, nb_classes, verbose)



