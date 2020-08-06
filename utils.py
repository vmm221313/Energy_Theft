import os
import numpy as np 
import pandas as po
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

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

class save_results():
	def __init__(self, model_name, y_true, y_pred):
		super(save_results, self).__init__()

		self.load_df()

		self.model_name = model_name
		self.y_true = y_true
		self.y_pred = y_pred

		self.evalutate_metrics()
		self.save_df()

	def area_under_the_curve(self):
		return roc_auc_score(self.y_true, self.y_pred)

	def average_precision_score(self):
		return average_precision_score(self.y_true, self.y_pred)

	def accuracy_score(self):
		return accuracy_score(self.y_true, self.y_pred)

	def f1_score(self):
		return f1_score(self.y_true, self.y_pred, average='micro')


	def evalutate_metrics(self):
		metrics = {'Model' : self.model_name, 
					'AUC' : self.area_under_the_curve(), 
					'MAP' : self.average_precision_score(), 
					'Accuracy' : self.accuracy_score(),
					'F1' : self.f1_score()
					}

		self.results_df = self.results_df.append(metrics, ignore_index=True)

	def load_df(self):
		if not os.path.exists('results/results.csv'):
			self.results_df = po.DataFrame(columns = ['Model'])
		else:
			self.results_df = po.read_csv('results/results.csv')

	def save_df(self):
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



