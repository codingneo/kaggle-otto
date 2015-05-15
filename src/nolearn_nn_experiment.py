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
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

from util import load_data, preprocess_data, preprocess_labels

'''
    Neural Net Using nolearn package
'''

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

def build_neuralnet(dims, nb_classes, n_hidden, nepoch):
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
        hidden1_num_units=1400,
        dropout_p=0.25,
        hidden2_num_units=n_hidden,
        dropout2_p=0.25,
        output_num_units=nb_classes,
        
        output_nonlinearity=softmax,

        #update=nesterov_momentum,
        update=adagrad,
        update_learning_rate=0.01,
        #update_momentum=0.9, only used with nesterov_
        eval_size=0.01,
        verbose=1,
        max_epochs=nepoch
    )

    return nn_model


def evaluate_neuralnet(X, y, dims, nb_classes, nfold):
    print("Building model...")

    scv = StratifiedKFold(y, nfold)

    best_score = float("inf")
    for n_hidden in range(200, 550, 50):
        nn_model = build_neuralnet(dims, nb_classes, n_hidden, 50)
        score = -cross_val_score(nn_model, X, y, cv=scv, scoring="log_loss")
        print("[" + str(n_hidden) + "] Average score is : " + str(score.mean()))

        if score.mean() < best_score:
            best_score = score.mean()

    return best_score


if __name__ == "__main__":
    print("Loading data...")
    np.random.seed(13251) # for reproducibility
    X, labels = load_data('./data/train.csv', train=True)
    X, scaler = preprocess_data(X)
    y, encoder = preprocess_labels(labels)

    nb_classes = len(encoder.classes_)
    print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')

    avg_logloss = evaluate_neuralnet(X, y, dims, nb_classes, 5)
    print("Average log loss of this neural network: " + str(avg_logloss))


    # X_test, ids = load_data('./data/test.csv', train=False)
    # X_test, _ = preprocess_data(X_test, scaler)

    # print("Training model...")
    # nn_model.fit(X, y)

    # proba = nn_model.predict_proba(X_test)
    # fs = './submission/nolearn/pickle_nolearn_' + str(seed)
    # with open(fs, 'w') as f:
    #     pickle.dump(proba, f)


    # print("Generating submission...")
    # model_file = './submission/nolearn/nolearn-otto-' + str(seed) + '.csv'
    # make_submission(proba, ids, encoder, fname=model_file)
