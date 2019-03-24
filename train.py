# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from time import time
from keras import backend as K
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import utils
from keras.wrappers.scikit_learn import KerasClassifier
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 3
np.random.seed(seed)

epochs = 5
batch_size = 128
n_iters = 10

X_train, y_train, X_test, y_test = utils.load_combined_data()
num_samples = len(X_train)
num_classes = y_test.shape[1]

def build_model(num_filters_0, filter_size_0, num_filters_1, filter_size_1, dropout, num_neurons):
	# create model
	inputs = Input(shape=(1, 28, 28))
	x = BatchNormalization(axis=3)(inputs)
	x = Conv2D(num_filters_0, (filter_size_0, filter_size_0), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Conv2D(num_filters_1, (filter_size_1, filter_size_1), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropout)(x)
	x = Flatten()(x)
	x = Dense(num_neurons, activation='relu')(x)
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(inputs, x)
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def search():
	params = {
		'num_filters_0': [12, 24, 36, 48],
		'filter_size_0': [3, 5, 7],
		'num_filters_1': [12, 24, 36, 48],
		'filter_size_1': [3, 5, 7],
		'dropout': uniform(0.0, 1.0),
		'num_neurons': randint(20, 200)
	}
	model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch_size)
	random_search = RandomizedSearchCV(estimator=model,
									   param_distributions=params,
									   n_iter=n_iters,
									   n_jobs=1)
	result = random_search.fit(X_train, y_train)

	print("Best: %f using %s" % (result.best_score_, result.best_params_))
	means = result.cv_results_['mean_test_score']
	stds = result.cv_results_['std_test_score']
	params = result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

def train():
	#params = {'dropout': 0.3253061838659205, 'filter_size_0': 5, 'filter_size_1': 7, 'num_filters_0': 12, 'num_filters_1': 48, 'num_neurons': 185}
	#params = {'dropout': 0.24249739821721294, 'filter_size_0': 7, 'filter_size_1': 5, 'num_filters_0': 48, 'num_filters_1': 12, 'num_neurons': 198}
	params = {'dropout': 0.1, 'filter_size_0': 5, 'filter_size_1': 5, 'num_filters_0': 24, 'num_filters_1': 36, 'num_neurons': 400}
	model = build_model(**params)
	# tensorboard for logging
	tensorboard = TensorBoard(log_dir='tensorboard/recognizer_{}'.format(num_samples))
	# Fit the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[tensorboard])
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print('Baseline Error: {}'.format(100-scores[1]*100))
	model.save('recognizer.h5')

#search()
train()