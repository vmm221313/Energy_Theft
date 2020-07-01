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


		



