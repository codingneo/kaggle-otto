import numpy as np
import pandas as pd
import pickle

from util import load_data, preprocess_data, preprocess_labels, make_submission

# bagging
# nolearn neural networks
fs1 = './submission/pickle_nolearn_13251'
fs2 = './submission/pickle_nolearn_25425'
fs3 = './submission/pickle_nolearn_35234'
fs4 = './submission/pickle_nolearn_4823'
fs5 = './submission/pickle_nolearn_96235'
fs6 = './submission/pickle_nolearn_74016'

# extreme gradient boosted trees
fs7 = './submission/pickle_xgbt-700-0.1-0.5-0.5-0.5'
fs8 = './submission/pickle_xgbt-700-0.3-0.9-0.8-0.2'

# keras neural networks
fs9 = './submission/pickle_keras_13251'

with open(fs1) as f:
	p1 = pickle.load(f)
with open(fs2) as f:
	p2 = pickle.load(f)
with open(fs3) as f:
	p3 = pickle.load(f)
with open(fs4) as f:
	p4 = pickle.load(f)
with open(fs5) as f:
	p5 = pickle.load(f)
with open(fs6) as f:
	p6 = pickle.load(f)
with open(fs7) as f:
	p7 = pickle.load(f)
with open(fs8) as f:
	p8 = pickle.load(f)
with open(fs9) as f:
	p9 = pickle.load(f)


# linear average
pred =0.35*(1.0/6.0*p1+1.0/6.0*p2+1.0/6.0*p3+1.0/6.0*p4+1.0/6.0*p5+1.0/6.0*p6) + \
			0.35*(p9) + 0.3*(0.5*p7+0.5*p8)

print("Loading data...")
X, labels = load_data('./data/train-expand.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('./data/test-expand.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

print("Generating submission...")
make_submission(pred, ids, encoder, fname='./submission/xgbt-nolearn-keras-stacking-new.csv')