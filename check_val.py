import argparse
import numpy as np
import pandas as pd
import os, pickle, sys

H_layer = [-1]
A_layer = [-1]
W_layer = []
B_layer = []
L = -1
activation = 'sigmoid'


def relu(x):
	return (x*(x>0))
def sigmoid(x): 
	# return 1.0 / (1.0 + np.exp(-x))
	return 0.5 * (1 + tanh(0.5*x))
def tanh(x):
	# return ( np.exp(x) - np.exp(-x) ) /  ( np.exp(x) + np.exp(-x) )
	return np.tanh(x)

def softmax(x):
  # e_x = np.exp(x)
  e_x = np.exp(x - np.max(x))
  out = e_x / e_x.sum()
  return out

def elu(x):
	return  x*(x >= 0) + (np.exp(x) - 1)*(x < 0)

def activation_func(x):
	if(activation == 'sigmoid'):
		return sigmoid(x)
	elif(activation == 'relu'):
		return relu(x)
	elif(activation == 'elu'):
		return elu(x)
	else:
		return tanh(x)

def forward_propagation(x):
	H_layer[0] = x
	for i in xrange(1, L):
		A_layer[i] = B_layer[i] + np.matmul(W_layer[i], H_layer[i-1])
		H_layer[i] = activation_func(A_layer[i])
	A_layer[L] = B_layer[L] + np.matmul(W_layer[L], H_layer[L-1])
	y_hat = softmax(A_layer[L]) 
	return y_hat

def pred_labels(test_path, theta_path):
	global H_layer, A_layer, W_layer, B_layer, num_features	

	t_df_test = pd.read_csv(test_path)
	t_df_test.set_index('id', inplace=True)
	
	num_features = len(t_df_test.columns) - 1
	# normalize between 0 and 1
	t_df_test.iloc[:, 0:num_features] /= 255.0
	X_test = t_df_test.iloc[:, 0:num_features].as_matrix()
	X_test = X_test.reshape([X_test.shape[0],X_test.shape[1],1])
	Y_test = t_df_test.iloc[:, num_features].as_matrix()
	id_list = t_df_test.index.values.tolist()

	theta = []
	ct = 0
	with open(theta_path, 'r') as f:
		while True:
			try:
				theta = pickle.load(f)
			except EOFError:
				break;
			print(ct)
			if(ct == 16):
				break
			ct += 1

	W_layer = theta[0]
	B_layer = theta[1]

	# for i in range(1, len(W_layer)):
	# 	print W_layer[i].shape
	
	# for i in range(1, len(B_layer)):
	# 	print B_layer[i].shape

	# sys.exit(0)

	for i in xrange(1, L+1):
		A_layer.append(np.zeros([B_layer[i].shape[0],1]))
	for i in xrange(1, L):
		H_layer.append(np.zeros([B_layer[i].shape[0],1]))


	data_dir = os.path.dirname(test_path)
	labels_path = os.path.join(data_dir, "predicted_labels.csv")

	labels_file = open(labels_path, "w", 1)
	labels_file.write("id,label\n")
	tot = X_test.shape[0]
	ct = 0.0

	for x_id, x, y in zip(id_list, X_test, Y_test):
		y_hat = forward_propagation(x)
		pred_class = np.argmax(y_hat)
		if(pred_class == y):
			ct += 1
		labels_file.write("{},{}\n".format(x_id, pred_class))
	labels_file.close()
	print('Accuracy: {}'.format(ct/tot))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", help="test path")
	parser.add_argument("--pickle_path", help="theta pickle path")
	parser.add_argument("--activation", help="activation function")
	parser.add_argument("--num_hidden", help="num_hidden")

	args = parser.parse_args()

	if(args.test):
		test_path = str(args.test)
	if (args.activation):
		activation = str(args.activation)
	if (args.pickle_path):
		theta_path = str(args.pickle_path)
	if (args.num_hidden):
		L = int(args.num_hidden) + 1

	pred_labels(test_path, theta_path)
