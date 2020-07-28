"""
This script will generate processed data for training using a variety of preprocessing methods
"""


import argparse

import numpy as np
import pandas as po
from tqdm import tqdm
from math import isinf 
import matplotlib.pyplot as plt
from numpy import array, zeros, full, argmin, inf, ndim

from filepaths import fp



def raw_impute():
	"""
	Takes the raw consumption data and makes the dataframe for imputation.
	In this df, rows with more than 30% missing cols in the 0 class are discarded.
	"""
	print('Taking raw consumption data and making the dataframe for imputation...')

	df = po.read_csv(fp.raw_data).sample(frac=1).reset_index(drop = True) # shuffle df so that classes are evenly distributed in train-val-test
	df = df.drop(['CONS_NO'], axis = 1)

	num_zeros = []
	for i in tqdm(range(len(df))):
		num_zeros.append(df.iloc[i][1:].to_list().count(0))
	df['num_zeros'] = num_zeros

	df_0 = df[df['FLAG'] == 0].reset_index(drop=True) #.sort_values('num_zeros')[:3615]
	df_1 = df[df['FLAG'] == 1].reset_index(drop=True)

	df_0 = df_0[df_0['num_zeros'] < 0.3*len(df_0.columns)]

	df_imp = po.concat([df_0, df_1], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)
	df_imp.to_csv(fp.imputation_raw, index=False)

	print('raw data for imputation ready, saved at {}'.format(fp.imputation_raw))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-funk', type=str, default=None)
	args = parser.parse_args()

	if args.funk != None:
		eval(args.funk+'()')

	#main(args)