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
from sklearn.cross_validation import StratifiedKFold

'''
    Neural Net Using nolearn package
'''

np.random.seed(13251) # for reproducibility

# def make_submission(y_prob, ids, encoder, fname):
#     with open(fname, 'w') as f:
#         f.write('id,')
#         f.write(','.join([str(i) for i in encoder.classes_]))
#         f.write('\n')
#         for i, probs in zip(ids, y_prob):
#             probas = ','.join([i] + [str(p) for p in probs.tolist()])
#             f.write(probas)
#             f.write('\n')
#     print("Wrote submission to file {}.".format(fname))

def build_neuralnet(dims, nb_classes):
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
        max_epochs=num_epochs
    )

    return nn_model


def nn_classify(shuffle_seed, num_epochs):
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
    nn_model = build_neuralnet(dims, nb_classes)

    print("Training model...")
    nn_model.fit(X, y)

    proba = nn_model.predict_proba(X_test)
    fs = './submission/nolearn/pickle_nolearn_' + str(seed)
    with open(fs, 'w') as f:
        pickle.dump(proba, f)


    print("Generating submission...")
    model_file = './submission/nolearn/nolearn-otto-' + str(seed) + '.csv'
    make_submission(proba, ids, encoder, fname=model_file)


for idx in range(60):
    seed = randint(0,10000)
    nepoch = 150
    nn_classify(seed, nepoch)
