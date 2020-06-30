import os
import numpy as np
from classifiers.fcn import Classifier_FCN

with open('data/train_splits_undersampled.npz', 'rb') as load_file:
    npzfile = np.load(load_file)    
    
    X_train = npzfile['X_train']
    X_val = npzfile['X_val']
    X_test = npzfile['X_test']
    
    y_train = npzfile['y_train']
    y_val = npzfile['y_val']
    y_test = npzfile['y_test']

outputs_dir = 'results/fcn/'

os.makedirs('results/', exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

fcn = Classifier_FCN(output_directory=outputs_dir, input_shape=X_train.shape[1:], nb_classes=2, verbose=True)

fcn.fit(X_train, y_train, X_val, y_val)
