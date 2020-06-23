import numpy as np 

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

