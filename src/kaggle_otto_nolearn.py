from __future__ import absolute_import
from __future__ import print_function

from random import randint

import numpy as np
import pandas as pd
import pickle

from keras.utils import np_utils, generic_utils
import lasagne
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    Neural Net Using nolearn package
'''

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    # if not scaler:
    #     scaler = StandardScaler()
    #     scaler.fit(X)
    # X = scaler.transform(X)

    # X = np.log(X+1.0)


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


def nn_classify(seed):
    np.random.seed(seed) # for reproducibility
    
    print("Loading data...")
    X, labels = load_data('./data/train.csv', train=True)
    X, scaler = preprocess_data(X)
    y, encoder = preprocess_labels(labels)

    X_test, ids = load_data('./data/test.csv', train=False)
    X_test, _ = preprocess_data(X_test, scaler)

    nb_classes = len(encoder.classes_)
    print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')

    print("Building model...")

    layers = [
        ('input', lasagne.layers.InputLayer),
        ('dropoutf', lasagne.layers.DropoutLayer),
        ('hidden1', lasagne.layers.DenseLayer),
        ('dropout', lasagne.layers.DropoutLayer),
        ('hidden2', lasagne.layers.DenseLayer),
        ('dropout2', lasagne.layers.DropoutLayer), 
        ('output', lasagne.layers.DenseLayer)
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
        max_epochs=150
    )

    print("Training model...")
    nn_model.fit(X, y)

    proba = nn_model.predict_proba(X_test)
    fs = './submission/nolearn/new-30/pickle_nolearn_' + str(seed)
    with open(fs, 'w') as f:
        pickle.dump(proba, f)


    print("Generating submission...")
    model_file = './submission/nolearn/new-30/nolearn-otto-' + str(seed) + '.csv'
    make_submission(proba, ids, encoder, fname=model_file)


for idx in range(30):
    seed = randint(0,10000)
    nn_classify(seed)
