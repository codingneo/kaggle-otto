from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle

from keras.utils import np_utils, generic_utils
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    Neural Net Using nolearn package
'''

np.random.seed(13251) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, original_labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        labels = original_labels.copy()
        X_2_3_4 = X[(labels=='Class_2') | (labels=='Class_3') | (labels=='Class_4')]
        labels_2_3_4 = labels[(labels=='Class_2') | (labels=='Class_3') | (labels=='Class_4')]
        labels[(labels=='Class_2') | (labels=='Class_3') | (labels=='Class_4')] = 'Class_2_3_4'
        return X, original_labels, labels, X_2_3_4, labels_2_3_4
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    # if not scaler:
    #     scaler = StandardScaler()
    #     scaler.fit(X)
    # X = scaler.transform(X)

    # X = np.log(X+1.0)

    # relabel the class 2, 3, 4 into a single class

    # normalized_X = X/np.tile(X.sum(axis=1).reshape(X.shape[0],1), (1, X.shape[1]))
    # log_transformed_X = np.log(X+1.0)
    # X = np.hstack((X, log_transformed_X, normalized_X))
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
    y = encoder.fit_transform(y).astype(np.int32)
    # if categorical:
    #     y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, original_labels, labels, X_2_3_4, labels_2_3_4 = load_data('./data/train.csv', train=True)
X, scaler = preprocess_data(X)
original_y, original_encoder = preprocess_labels(original_labels)
y, encoder = preprocess_labels(labels)
X_2_3_4, scaler_2_3_4 = preprocess_data(X_2_3_4)
y_2_3_4, encoder_2_3_4 = preprocess_labels(labels_2_3_4)


X_test, ids = load_data('./data/test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = len(encoder.classes_)
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

layers = [
    ('input', layers.InputLayer),
    ('dropoutf', layers.DropoutLayer),
    ('hidden1', layers.DenseLayer),
    ('dropout', layers.DropoutLayer),
    ('hidden2', layers.DenseLayer),
    ('dropout2', layers.DropoutLayer), 
    ('output', layers.DenseLayer)
]

nn_model = NeuralNet(
    layers=layers,
    input_shape=(None, dims),
    dropoutf_p=0.15,
    hidden1_num_units=1000,
    dropout_p=0.25,
    hidden2_num_units=500,
    dropout2_p=0.25,
    output_num_units=nb_classes,
    
    output_nonlinearity=softmax,

    #update=nesterov_momentum,
    update=adagrad,
    update_learning_rate=0.01,
    #update_momentum=0.9, only used with nesterov_
    eval_size=0.01,
    verbose=1,
    max_epochs=50
)

print("Training layer 1 model...")
nn_model.fit(X, y)
proba = nn_model.predict_proba(X_test)

print("Training layer 2 model...")
nn_model_2_3_4 = NeuralNet(
    layers=layers,
    input_shape=(None, dims),
    dropoutf_p=0.3,
    hidden1_num_units=500,
    dropout_p=0.3,
    hidden2_num_units=250,
    dropout2_p=0.3,
    output_num_units=3,
    
    output_nonlinearity=softmax,

    #update=nesterov_momentum,
    update=adagrad,
    update_learning_rate=0.01,
    #update_momentum=0.9, only used with nesterov_
    eval_size=0.01,
    verbose=1,
    max_epochs=50
)

nn_model_2_3_4.fit(X_2_3_4, y_2_3_4)
proba_2_3_4 = nn_model_2_3_4.predict_proba(X_test)

class_2_3_4 = encoder.transform(['Class_2_3_4'])[0]
probs_2_3_4 = np.multiply(np.tile(proba[:,class_2_3_4].reshape(proba.shape[0],1),(1,3)), proba_2_3_4)

final_proba = np.ndarray(shape=(proba.shape[0],len(original_encoder.classes_)), dtype=float)
for i in original_encoder.classes_:
    idx = original_encoder.transform([str(i)])[0]
    if (str(i) != 'Class_2' and str(i) != 'Class_3' and str(i) != 'Class_4'):
        final_proba[:,idx] = proba[:,encoder.transform([str(i)])[0]]
    else:
        final_proba[:,idx] = probs_2_3_4[:,encoder_2_3_4.transform([str(i)])[0]]

fs = './submission/pickle_nolearn_13251'
with open(fs, 'w') as f:
    pickle.dump(proba, f)


print("Generating submission...")
make_submission(final_proba, ids, original_encoder, fname='./submission/nolearn-otto-13251-test.csv')

