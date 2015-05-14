import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

import graphlab as gl
gl.product_key.set_product_key('256B-A11C-F81B-157F-69CA-048F-ABD9-17DA')

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
	"""Multi class version of Logarithmic Loss metric.
	https://www.kaggle.com/wiki/MultiClassLogLoss

	Parameters
	----------
	y_true : array, shape = [n_samples]
						true class, intergers in [0, n_classes - 1)
	y_pred : array, shape = [n_samples, n_classes]

	Returns
	-------
	loss : float
	"""
	predictions = np.clip(y_pred, eps, 1 - eps)

	# normalize row sums to 1
	predictions /= predictions.sum(axis=1)[:, np.newaxis]

	actual = np.zeros(y_pred.shape)
	n_samples = actual.shape[0]
	actual[np.arange(n_samples), y_true.astype(int)] = 1
	vectsum = np.sum(actual * np.log(predictions))
	loss = -1.0 / n_samples * vectsum

	return loss


def make_submission(y_prob, ids, encoder, fname):
	with open(fname, 'w') as f:
		f.write('id,')
		f.write(','.join([str(i) for i in encoder.classes_]))
		f.write('\n')
		for i, probs in zip(ids, y_prob):
			probas = ','.join([str(i)] + [str(p) for p in probs.tolist()])
			f.write(probas)
			f.write('\n')

			print("Wrote submission to file {}.".format(fname))


train_url = './data/train.csv'
train = gl.SFrame.read_csv(train_url)
test_url = './data/test.csv'
test = gl.SFrame.read_csv(test_url)

# Make a train-test split
train.remove_column('id')
train_data, valid_data = train.random_split(0.85)
# train_data.remove_column('id')
# valid_data.remove_column('id')

# test.remove_column('id')


encoder = LabelEncoder()
encoder.fit(valid_data['target'])
valid_y = encoder.transform(valid_data['target']).astype(np.int32)

# Create a model with validation.
best_logloss = float("inf")
best_params = []
for ssize in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
	model = gl.boosted_trees_classifier.create(train_data, target='target',
	                                           max_iterations=1100, 
	                                           validation_set=valid_data,
	                                           step_size=ssize,#0.02,
	                                           min_child_weight=2.0,
	                                           max_depth=12,
	                                           row_subsample=0.5, column_subsample=0.5)


	# Save predictions to an SFrame (class and corresponding class-probabilities)
	valid_pred = model.predict_topk(valid_data, k=9)
	valid_pred['id'] = valid_pred['id'].apply(lambda x: int(x))
	valid_pred = valid_pred.sort(['id','class'])


	logloss = multiclass_log_loss(np.asarray(valid_y),np.asarray(valid_pred['probability']).reshape((valid_data.shape[0],9)))

	if (logloss < best_logloss):
		best_logloss = logloss
		best_params = ssize

		full_model = gl.boosted_trees_classifier.create(train, target='target',
                                           					max_iterations=1100, 
                                           					validation_set=None,
                                           					step_size=best_params,
                                           					min_child_weight=2.0,
                                           					max_depth=12,
                                           					row_subsample=0.5, column_subsample=0.5)


		test_pred = full_model.predict_topk(test, k=9)
		test_pred['id'] = test_pred['id'].apply(lambda x: int(x))
		test_pred = test_pred.sort(['id','class'])
		test_probs = np.asarray(test_pred['probability']).reshape((test.shape[0],9))


		fs = './submission/graphlab-xgbt/pickle_xgbt-' + str(1100) + '_' + str(12) + '_' + str(best_params) + '-' + str(best_logloss)
		with open(fs, 'w') as f:
			pickle.dump(test_probs, f)

# model = gl.boosted_trees_classifier.create(train, target='target',
#                                            max_iterations=best_params[0], 
#                                            validation_set=None,
#                                            step_size=0.02,
#                                            min_child_weight=0.5,
#                                            max_depth=best_params[1],
#                                            row_subsample=0.5, column_subsample=0.5)

# test_pred = model.predict_topk(test, k=9)
# test_pred['id'] = test_pred['id'].apply(lambda x: int(x))
# test_pred = test_pred.sort(['id','class'])
# test_probs = np.asarray(test_pred['probability']).reshape((test.shape[0],9))

# submission_file = './submission/graphlab-xgbt-otto' + str(best_params[0]) + '-' + str(best_params[1]) + '.csv'
# make_submission(test_probs, test['id'], encoder, submission_file)

