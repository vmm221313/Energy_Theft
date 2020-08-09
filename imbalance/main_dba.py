import numpy as np
import pandas as po
import matplotlib.pyplot as plt

from utils.augment import augment_train_set

def main():
    df = po.read_csv('../data/processed/imputation/edtwbi.csv').sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop(['num_zeros', 'FLAG'], axis=1)
    y = df['FLAG']  

    X_train = X[:int(0.6*len(X))]
    y_train = y[:int(0.6*len(X))]

    X_val   = X[int(0.6*len(X)):int(0.8*len(X))]
    y_val   = y[int(0.6*len(X)):int(0.8*len(X))]

    X_test  = X[int(0.8*len(X)):]
    y_test  = y[int(0.8*len(X)):]

    X_train     = X_train.to_numpy()[:, :, np.newaxis]
    X_val       = X_val.to_numpy()[:, :, np.newaxis]
    X_test      = X_test.to_numpy()[:, :, np.newaxis]

    y_train     = y_train.to_numpy()
    y_val       = y_val.to_numpy()
    y_test      = y_test.to_numpy()

    N       = 100 

    classes, classes_counts = np.unique(y_train, return_counts=True)    

    # augment the dataset
    X_train_synth, y_train_synth = augment_train_set(X_train, y_train, classes, N, limit_N = False)

if __name__ == '__main__':
    main()