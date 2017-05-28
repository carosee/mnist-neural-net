import scipy.io
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from timeit import default_timer as timer


# properties of neural network
NHID = 200 # hidden layer size
NIN = 784 # number of inputs = number of features of digits dataset
NOUT = 10 # number of outputs = number of classes


# math functions
def sigmoid(x):
	return 1/(1+np.exp(-x))

def dsigmoid(x):
	return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def dtanh(x):
	return 1 - (tanh(x) ** 2)


# data preprocessing
def array_bias(arr):
	return np.insert(arr, len(arr), 1)

def matrix_bias(mat):
	mat = mat.tolist()
	for elem in mat:
		elem.insert(len(mat), 1)
	return np.array(mat)

def partition_data(data, labels, num_training):
	num_validation = len(data) - num_training

	shuffled = np.random.permutation(len(data))
	validation_set = shuffled[:num_validation]
	training_set = shuffled[num_validation : num_training+num_validation]

	training_data = []
	training_labels = []

	for i in range(num_training):
		index = training_set[i]
		training_data.append(data[index])
		training_labels.append(labels[index])

	validation_data = []
	validation_labels = []

	for i in range(num_validation):
		index = validation_set[i]
		validation_data.append(data[index])
		validation_labels.append(labels[index])

	return [np.array(training_data), np.array(training_labels), np.array(validation_data), np.array(validation_labels)]

def scale_data(data, upper=3.0, lower=0.0):
	mi, ma = np.min(data, axis=0), np.max(data, axis=0)
	return upper - (((upper-lower) * (ma-data)) / (ma-mi))

def clip_values(vals):
	vals[np.isinf(vals)] = 0
	vals[np.isneginf(vals)] = 0
	vals[np.isnan(vals)] = 0
	return vals

def plot_data(accuracies, title):
    it = [ i+1 for i in range(len(accuracies)) ]
    plt.plot(it, accuracies, '-o')
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.show()


# error and loss fns
def mean_squared_error(predicted, actual):
	diff = np.subtract(predicted, actual)
	return 0.5 * np.sum(np.square(diff))

def dmean_squared_error(predicted, actual):
	return -1 * np.subtract(actual, predicted)

def cross_entropy_error(predicted, actual):
	l1 = np.log(predicted.clip(min=0.01))
	l2 = np.log((1-predicted).clip(min=0.01))
	return -1 * np.sum(l1 * actual + l2 * (1-actual))

def dcross_entropy_error(predicted, actual):
	return -1 * (clip_values(actual/predicted) - clip_values((1-actual)/(1-predicted)))


# neural network algs
def forward(w1, w2, image):
	image = array_bias(image)
	z2 = np.ravel(np.dot(image, w1))
	a2 = array_bias(tanh(np.array(z2).ravel()))
	z3 = np.ravel(np.dot(a2, w2))
	yHat = sigmoid(np.array(z3).ravel())
	return (w1, w2, image, z2, a2, z3, yHat)

def backpropagate_mse(image, label, forward_params):
	(w1, w2, image, z2, a2, z3, yHat) = forward_params
	actual = np.zeros(10)
	actual[label] = 1
	loss = mean_squared_error(yHat, actual)
	e = dmean_squared_error(yHat, actual)
	delta3 = np.dot(e, np.diag(dsigmoid(z3))) # needs to be 10
	dJdw2 = np.dot(np.transpose(np.matrix(a2)), np.matrix(delta3))
	xy = np.dot(np.transpose(np.matrix(image)), np.matrix(delta3))
	xyw = np.dot(xy, w2.T) #785x201
	xyw = np.delete(xyw, len(xyw[0])-1, axis=1)
	dJdw1 = np.dot(xyw, np.diag(dtanh(z2)))
	return dJdw1, dJdw2, loss

def backpropagate_cee(image, label, forward_params):
	(w1, w2, image, z2, a2, z3, yHat) = forward_params
	actual = np.zeros(10)
	actual[label] = 1
	loss = cross_entropy_error(yHat, actual)
	e = dcross_entropy_error(yHat, actual)
	delta3 = np.dot(e, np.diag(dsigmoid(z3))) # needs to be 10
	dJdw2 = np.dot(np.transpose(np.matrix(a2)), np.matrix(delta3))
	xy = np.dot(np.transpose(np.matrix(image)), np.matrix(delta3))
	xyw = np.dot(xy, w2.T) #785x201
	xyw = np.delete(xyw, len(xyw[0])-1, axis=1)
	dJdw1 = np.dot(xyw, np.diag(dtanh(z2)))
	return dJdw1, dJdw2, loss


def train(images, labels, epsilon, epochs, loss='mse'):
	if loss == 'cee':
		backpropagate = backpropagate_cee
	else:
		backpropagate = backpropagate_mse
	# initialize weights at random and do more stuff
	r1 = np.random.normal(loc=0.0, scale=0.02, size=NIN*(NHID))
	r2 = np.random.normal(loc=0.0, scale=0.02, size=NHID*NOUT)
	b1, b2 = np.array([np.ones(NHID)]), np.array([np.ones(NOUT)])
	w1 = np.concatenate((np.reshape(r1, (NIN, NHID)), b1), axis=0)
	w2 = np.concatenate((np.reshape(r2, (NHID, NOUT)), b2), axis=0)
	# error = []
	# gradient descent iterations
	accs = []
	for epoch in range(1, epochs+1):
		n = np.random.choice(len(images), len(images), replace=False)
		image_subset, label_subset = images[n], labels[n]
		for i in range(len(images)):
			image, label = image_subset[i], label_subset[i]
			params = forward(w1, w2, image)
			dJdw1, dJdw2, loss = backpropagate(image, label, params)
			w1 = w1 - (epsilon/epoch) * dJdw1
			w2 = w2 - (epsilon/epoch) * dJdw2
			print("epoch", epoch, "iteration", i, "loss", loss)
			# error.append(loss)
			if i % 1000 == 0:
				p = predict(validation_data, w1, w2)
				error = float(np.sum(p != validation_labels)) / len(p)
				accs.append(error)
	return accs, w1, w2

def predict(images, w1, w2):
	images = matrix_bias(images)
	z2 = np.matrix(images) * w1
	a2 = matrix_bias(tanh(z2))
	z3 = np.matrix(a2) * w2
	yHat = sigmoid(z3)
	return np.array(np.argmax(yHat, axis=1)).ravel()

def run_MSE():
	start = timer()
	mse_accuracies, w1, w2 = train(training_data, training_labels, 0.01, 20, loss='mse')
	time = timer() - start
	print("MSE Accuracies", mse_accuracies)
	print("MSE training took", time)
	plot_data(mse_accuracies, 'Mean Squared Loss Function')

def run_CEE():
	start = timer()
	cee_accuracies, w1, w2 = train(training_data, training_labels, 0.0001, 20, loss="cee")
	time = timer() - start
	print("CEE Accuracies", cee_accuracies)
	print("CEE training took", time)
	plot_data(cee_accuracies, 'Cross Entropy Loss Function')


# write csv for Kaggle
def write_csv():
	testmat = scipy.io.loadmat('dataset/test.mat')
	test_data = testmat['test_images']
	reshaped_testing = np.ndarray((10000, 784))
	# plt.imshow(test_data[3])
	# plt.show()
	for i in range(10000):
		new = np.ravel(test_data[i])
		reshaped_testing[i] = new
	test_data = clip_values(scale_data(reshaped_testing))

	start = timer()
	mse_accuracies, w1, w2 = train(training_data, training_labels, 0.01, 20, loss='mse')
	time = timer() - start
	print("MSE Accuracies", mse_accuracies)
	print("MSE training took", time)

	predictions = predict(test_data, w1, w2)
	c = csv.writer(open("digits.csv", "w", newline=""))
	c.writerow(["Id","Category"])
	for i in range(len(predictions)):
		c.writerow([i+1,int(predictions[i])])


# load data
mat = scipy.io.loadmat('dataset/train.mat')
training = mat['train_images'] #28,28,60000
data = training.reshape(784, 60000).swapaxes(0,1)
labels = np.ravel(mat['train_labels'])
data = clip_values(scale_data(data))


# preprocess data
partitioned = partition_data(data, labels, 50000)
training_data = partitioned[0]
training_labels = partitioned[1]
validation_data = partitioned[2]
validation_labels = partitioned[3]


# run stuff here
run_MSE()
run_CEE()
write_csv()
