import os
import pickle

import numpy as np
import pandas as pd

from util import load_data, preprocess_data, preprocess_labels, make_submission

# bagging
# nolearn neural networks
path = "./submission/nolearn/30/"
files = os.listdir(path)

ps = []
for file in files:
	if "pickle_nolearn" in file:
		with open(path+str(file)) as f:
			p = pickle.load(f)
		ps.append(p)

gbt_fs = './submission/graphlab-xgbt/pickle_xgbt-900_18-0.436780769313'
with open(gbt_fs) as f:
	gbt_p = pickle.load(f)

print("Loading data...")
X, labels = load_data('./data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('./data/test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

print(len(encoder.classes_))
pred = np.ndarray(shape=(X_test.shape[0], len(encoder.classes_)), dtype=float)
for p in ps:
	pred += p
pred /= len(ps)

pred = 0.7*pred + 0.3*gbt_p

print("Generating submission...")
make_submission(pred, ids, encoder, fname='./submission/nolearn-30-gbt-stacking.csv')