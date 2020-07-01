import os
import numpy as np
from utils import save_results
from classifiers.fcn import Classifier_FCN
from classifiers.resnet import Classifier_RESNET

with open('data/train_splits_undersampled.npz', 'rb') as load_file:
    npzfile = np.load(load_file)    
    
    X_train = npzfile['X_train']
    X_val = npzfile['X_val']
    X_test = npzfile['X_test']
    
    y_train = npzfile['y_train']
    y_val = npzfile['y_val']
    y_test = npzfile['y_test']

model_name = 'resnet'

outputs_dir = 'results/{}/'.format(model_name)

os.makedirs('results/', exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

classifier = Classifier_RESNET(output_directory=outputs_dir, input_shape=X_train.shape[1:], nb_classes=2, verbose=True)
classifier.fit(X_train, y_train, X_val, y_val)

y_pred = classifier.predict(X_test)

'''
print(y_pred.shape)
print(y_test.shape)
print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_test)
print(np.argmax(y_test, axis = 1))
'''

save_results(model_name, np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

#resnet = Classifier_RESNET(output_directory=outputs_dir, input_shape=X_train.shape[1:], nb_classes=2, verbose=True)
#resnet.fit(X_train, y_train, X_val, y_val)