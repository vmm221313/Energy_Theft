"""
eDTWBI Imputation
"""
import pickle
import argparse
import numpy as np
import pandas as po
from tqdm import tqdm
from math import isinf 
import multiprocessing
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy import array, zeros, full, argmin, inf, ndim

from filepaths import fp

def _traceback(D):
	i, j = array(D.shape) - 2
	p, q = [i], [j]
	while (i > 0) or (j > 0):
		tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
		if tb == 0:
			i -= 1
			j -= 1
		elif tb == 1:
			i -= 1
		else:  # (tb == 2):
			j -= 1
		p.insert(0, i)
		q.insert(0, j)
	return array(p), array(q)

def derivative_dtw_distance(i, j, x, y):
	if i+1 == len(x) or j+1 == len(y):
		dist = (x[i] - y[j])**2
	
	else:
		d_x_i = ((x[i] - x[i-1]) + (x[i+1] - x[i-1])/2)/2
		d_y_j = ((y[j] - y[j-1]) + (y[j+1] - y[j-1])/2)/2    

		dist = (d_x_i - d_y_j)**2

	return dist

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
	"""
	Computes Dynamic Time Warping (DTW) of two sequences.
	:param array x: N1*M array
	:param array y: N2*M array
	:param func dist: distance used as cost measure
	:param int warp: how many shifts are computed.
	:param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
	:param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
	Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
	"""
	assert len(x)
	assert len(y)
	assert isinf(w) or (w >= abs(len(x) - len(y)))
	assert s > 0
	r, c = len(x), len(y)
	if not isinf(w):
		D0 = full((r + 1, c + 1), inf)
		for i in range(1, r + 1):
			D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
		D0[0, 0] = 0
	else:
		D0 = zeros((r + 1, c + 1))
		D0[0, 1:] = inf
		D0[1:, 0] = inf
	D1 = D0[1:, 1:]  # view
	for i in range(r):
		for j in range(c):
			if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
				D1[i, j] = dist(i, j, x, y)
	C = D1.copy()
	jrange = range(c)
	for i in range(r):
		if not isinf(w):
			jrange = range(max(0, i - w), min(c, i + w + 1))
		for j in jrange:
			min_list = [D0[i, j]]
			for k in range(1, warp + 1):
				i_k = min(i + k, r)
				j_k = min(j + k, c)
				min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
			D1[i, j] += min(min_list)
	if len(x) == 1:
		path = zeros(len(y)), range(len(y))
	elif len(y) == 1:
		path = range(len(x)), zeros(len(x))
	else:
		path = _traceback(D0)
	return D1[-1, -1], C, D1, path


def find_gaps(x):
	seq = False
	missing_seqs = []
	seq_start_idx = -1
	for i in range(len(x)):
		if seq == False and x[i] == 0:
			seq = True
			seq_start_idx = i
		
		elif seq == True and x[i] != 0:
			seq = False
			if seq_start_idx == -1:
				raise
			missing_seqs.append((seq_start_idx, i))
			seq_start_idx = -1
	
	return missing_seqs

def dtwbi(D, Q, len_gap, stride=1):
	
	min_dtw_cost = inf
	start_index = 0
	
	for i in range(0, len(D)-len_gap, stride):
		#print(i, i+len_gap)
		try:
			cost, cost_matrix, acc_cost_matrix, path = dtw(D[i:i+len_gap], Q, dist=derivative_dtw_distance)
		except:
			print(D[i*len_gap:(i+1)*len_gap])
			print(Q)
			print(len_gap)
			print(len(D))
			print(i*len_gap, (i+1)*len_gap)
			raise
		if cost < min_dtw_cost:
			min_dtw_cost = cost
			start_index=i

	return start_index

def apply_dtwbi_after(x, start_index, end_index):
	len_gap = end_index - start_index
	
	Qa = x[end_index:end_index+len_gap]
	Da = x[end_index+len_gap:]
	
	Qas_start = dtwbi(Da, Qa, len_gap)
	#Qas = x[Qas_start:Qas_start+len_gap]
	
	refA = x[Qas_start-len_gap:Qas_start] # Previous window of Qas
	if Qas_start-len_gap < 0:
		refA = x[Qas_start:Qas_start+len_gap] # = Qa
	
	return refA

def apply_dtwbi_before(x, start_index, end_index):
	len_gap = end_index - start_index
	
	Qb = x[start_index-len_gap:start_index]
	Db = x[:start_index-len_gap]
	
	Qbs_start = dtwbi(Db, Qb, len_gap)
	#Qbs = x[Qbs_start:Qbs_start+len_gap]
	
	refB = x[Qbs_start+len_gap:Qbs_start+2*len_gap] # Next window of Qbs
	if Qbs_start+2*len_gap > len(x):
		refB = x[Qbs_start:Qbs_start+len_gap] # = Qb
	
	return refB

def edtwbi(x, start_index, end_index):
	len_gap = end_index - start_index
	
	if end_index + len_gap > len(x):
		refB = apply_dtwbi_before(x, start_index, end_index) # only dtwbi in other direction
		
		return refB
		
	elif start_index-len_gap < 0:
		refA = apply_dtwbi_after(x, start_index, end_index) # only dtwbi in other direction
		
		return refA
	
	else: # both cannot simultaneously happen, so not keeping a case for that
		refA = apply_dtwbi_after(x, start_index, end_index)
		refB = apply_dtwbi_before(x, start_index, end_index)
		
		return np.mean([np.array(refA), np.array(refB)], axis = 0)

def impute_row(i):
	global df
	row = df.iloc[i][:-2].to_list()
	row_filled = row.copy()
	
	gaps = find_gaps(row)
	for gap in gaps:
		row_filled[gap[0]:gap[1]] = edtwbi(row, gap[0], gap[1])

	return row_filled

def main(args):
	global df
	df = po.read_csv(fp.imputation_raw)
	print('Read df')

	if args.run_mode == 'parallel':
		p = multiprocessing.Pool() 
		new_rows = list(tqdm(p.imap(impute_row, list(range(len(df)))), total=len(df)))

	elif args.run_mode == 'map':
		new_rows = list(tqdm(map(impute_row, list(range(len(df)))), total=len(df))) 

	elif args.run_mode == 'loop':
		new_rows = []
		for i in tqdm(range(len(df))):
			new_rows.append(impute_row(i))

	with open(fp.edtwbi_imputed, 'wb') as f:
		pickle.dump(new_rows, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-run_mode', type=str, default='map')
	args = parser.parse_args()

	main(args)
