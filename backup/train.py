# Libraries
import argparse
import sets
import numpy as np
import pandas as pd
import os, pickle
import copy
import sklearn.utils as sk

# interact with code with `kill -SIGUSR2 <PID>`
import code
import signal
signal.signal(signal.SIGUSR2, lambda sig, frame: code.interact(local=dict(globals(), **locals())))

# Declare the variables to be
# read from command line

# initial learning rate
lr = 0
# momentum
momentum = 0
# number of hidden layers
num_hidden = 0
# sizes of hidden layers
sizes = [784]
# activation function
activation = 'tanh'
# loss function
loss = 'ce'
# optimization algorithm
opt = 'gd'
# batch size
batch_size = 20
# annealing
anneal = False
# model save directory
save_dir = '/pa1'
# log save directory
expt_dir = '/pa1/exp'
# path to train dataset
train_path = 'train.csv'
# path to test dataset
test_path = 'test.csv'
# path to validation dataset
val_path = ''
# log frequency (steps)
log_frequency = 1375
# max epochs
max_epochs = 60
# anneal type
anneal_type = "val"
# anneal lr threshold
anneal_threshold = 1e-8

# neural network parameters
W_layer = [-1]
B_layer = [-1]
# neuron inputs and activations
A_layer = [-1]
H_layer = [-1]
# Total number of layers
L = 0
# list containing all parameters
theta = []

def main():
	global theta
	# read training data into a pandas dataframe
	t_df = pd.read_csv(train_path)
	t_df.set_index('id', inplace=True)
	num_features = len(t_df.columns)-1
	# import pdb
	# pdb.set_trace()
	# normalize between 0 and 1 - not required?
	t_df.iloc[:, 0:num_features] /= 255.0
	X = t_df.iloc[:, 0:num_features].as_matrix()
	X = X.reshape([X.shape[0],X.shape[1],1])
	Y_temp = t_df.iloc[:, num_features].as_matrix()
	Y = np.zeros([Y_temp.shape[0],10,1])
	for i in range(Y_temp.shape[0]):
		Y[i] = get_output_vector(Y_temp[i])


	global X_val, Y_val
	# read validation data into a pandas dataframe
	t_df_val = pd.read_csv(val_path)
	t_df_val.set_index('id', inplace=True)
	# normalize between 0 and 1 - not required?
	t_df_val.iloc[:, 0:num_features] /= 255.0
	X_val = t_df_val.iloc[:, 0:num_features].as_matrix()
	X_val = X_val.reshape([X_val.shape[0],X_val.shape[1],1])
	Y_val_temp = t_df_val.iloc[:, num_features].as_matrix()
	Y_val = np.zeros([Y_val_temp.shape[0],10,1])
	for i in range(Y_val_temp.shape[0]):
		Y_val[i] = get_output_vector(Y_val_temp[i])


	# # read test data into a pandas dataframe
	# t_df_test = pd.read_csv(test_path)
	# t_df_test.set_index('id', inplace=True)
	# # normalize between 0 and 1
	# t_df_test.iloc[:, 0:num_features] /= 255.0
	# X_test = t_df_test.iloc[:, 0:num_features].as_matrix()
	# X_test = X_test.reshape([X_test.shape[0],X_test.shape[1],1])

	
	# print(X.shape)
	# print(Y.shape)

	# neural network parameters
	# Use Xavier Glorot init for Weights - zero mean, suitable variance?
	for i in xrange(1, L+1):
		root = np.sqrt(6/(sizes[i] + sizes[i-1]))
		W_layer.append(np.random.uniform(-1*root, root,[sizes[i],sizes[i-1]]))
		# W_layer.append(np.random.randn(sizes[i],sizes[i-1]))
	# Initialize bias as zeros
	for i in xrange(1, L+1):
		# B_layer.append(np.zeros([sizes[i],1]))
		# B_layer.append(np.random.uniform(0,1,[sizes[i],1]))
		B_layer.append(np.random.randn(sizes[i],1))
	# neuron inputs and activations
	for i in xrange(1, L+1):
		A_layer.append(np.zeros([sizes[i],1]))
	for i in xrange(1, L):
		H_layer.append(np.zeros([sizes[i],1]))
	
	theta = init_theta()
	
	global log_train_file, log_val_file, theta_pickle_file
	log_train_file = open(os.path.join(expt_dir, "log_train.txt"), "w", 1)
	log_val_file = open(os.path.join(expt_dir, "log_val.txt"), "w", 1)
	theta_pickle_file = open(os.path.join(save_dir, "theta.pickle"), "w")

	if(opt == 'gd'):
		do_mini_batch_gradient_descent(X, Y)
	elif(opt == 'momentum'):
		momentum_gradient_descent(X, Y)
	elif(opt == 'nag'):
		nag_gradient_descent(X, Y)
	else:
		adam_gradient_descent(X, Y)

	# pickle.dump(theta, theta_pickle_file)
	log_train_file.close()
	log_val_file.close()
	theta_pickle_file.close()


if __name__ == "__main__":
	# Initialize the parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", help="learning rate")
	parser.add_argument("--momentum", help="momentum")
	parser.add_argument("--num_hidden", help="num_hidden")
	parser.add_argument("--sizes", help="sizes of hidden layers")
	parser.add_argument("--activation", help="activation function")
	parser.add_argument("--loss", help="loss function")
	parser.add_argument("--opt", help="optimization algorithm")
	parser.add_argument("--batch_size", help="batch_size")
	parser.add_argument("--anneal", help="anneal")
	parser.add_argument("--save_dir", help="save directory")
	parser.add_argument("--expt_dir", help="log directory")
	parser.add_argument("--train", help="train path")
	parser.add_argument("--test", help="test path")
	parser.add_argument("--val", help="validation path")
	args = parser.parse_args()

	# Process the command line arguments
	if(args.lr):
		lr = float(args.lr)
		if(args.momentum):
			momentum = float(args.momentum)
	if(args.num_hidden):
		num_hidden = int(args.num_hidden)
	if(args.sizes):
		tmp = args.sizes.split(',')
		if(len(tmp)!=num_hidden):
			print('argument mismatch!')
			exit()
		for x in tmp:
			sizes.append(int(x))
		sizes.append(10)
	if(args.activation):
		activation = str(args.activation)
		options = sets.Set(['sigmoid','tanh','relu'])
		if(activation not in options):
			print('Invalid activation function')
			exit()
	if(args.loss):
		loss = str(args.loss)
		options = sets.Set(['sq','ce'])
		if(loss not in options):
			print('Invalid loss function')
			exit()
	if(args.opt):
		opt = str(args.opt)
		options = sets.Set(['gd', 'momentum', 'nag', 'adam'])
		if(opt not in options):
			print('Invalid loss function')
			exit()
	if(args.batch_size):
		batch_size = int(args.batch_size)
		if(batch_size != 1 and batch_size%5 != 0):
			print('Invalid batch size')
			exit()
	if(args.anneal):
		if(args.anneal == 'true' or args.anneal == 'True'):
			anneal = True
	if(args.save_dir):
		save_dir = str(args.save_dir)
	if(args.expt_dir):
		expt_dir = str(args.expt_dir)
	if(args.train):
		train_path = str(args.train)
	if(args.test):
		test_path = str(args.test)
	if (args.val):
		val_path = str(args.val)
# Set the total number of layers
L = num_hidden + 1

def print_params():
	print("lr: {}".format(lr))
	print("momentum: {}".format(momentum))
	print("num_hidden: {}".format(num_hidden))
	print("sizes: {}".format(sizes))
	print("activation: {}".format(activation))
	print("loss: {}".format(loss))
	print("opt: {}".format(opt))
	print("batch_size: {}".format(batch_size))
	print("anneal: {}".format(anneal))
	print("save_dir: {}".format(save_dir))
	print("expt_dir: {}".format(expt_dir))
	print("train_path: {}".format(train_path))
	print("test_path: {}".format(test_path))
	print("val_path: {}".format(val_path))

def cross_entropy_loss(y_hat, y):
	return -1 * np.log2(y_hat[np.argmax(y)])[0]

def square_loss(y_hat, y):
	return np.sum(np.square(y - y_hat))

def get_loss(y_hat, y):
	if(loss == 'ce'):
		return cross_entropy_loss(y_hat, y)
	else:
		return square_loss(y_hat, y)

def get_output_vector(n):
	out = np.zeros([10,1])
	out[n][0] = 1
	return out

def sigmoid(x): 
	# return 1.0 / (1.0 + np.exp(-x))
	return 0.5 * (1 + tanh(0.5*x))

def sigmoid_der(x): 
	return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
	# return ( np.exp(x) - np.exp(-x) ) /  ( np.exp(x) + np.exp(-x) )
	return np.tanh(x)

def tanh_der(x):
	return (1 - tanh(x) ** 2 )

def relu(x):
	return (x*(x>0))

# check division by 0?
def relu_der(x):
	return ( (x*(x>0)) / (x+(x==0)) )

def activation_func(x):
	if(activation == 'sigmoid'):
		return sigmoid(x)
	elif(activation == 'relu'):
		return relu(x)
	else:
		return tanh(x)

def activation_der(x):
	if(activation == 'sigmoid'):
		return sigmoid_der(x)
	elif(activation == 'relu'):
		return relu_der(x)
	else:
		return tanh_der(x)

def softmax(x):
  # e_x = np.exp(x)
  e_x = np.exp(x - np.max(x))
  out = e_x / e_x.sum()
  return out

def calc_error_loss(X, Y):
	num_samples, num_correct, loss = X.shape[0], 0, 0
	true_positive_count = np.zeros(10)
	false_positive_count = np.zeros(10)
	false_negative_count = np.zeros(10)
	for x, y in zip(X, Y):
		y_hat = forward_propagation(x)
		loss += get_loss(y_hat, y)
		true_class = np.argmax(y)
		pred_class = np.argmax(y_hat)
		if (true_class == pred_class):
			num_correct += 1
			true_positive_count[true_class] += 1
		else:
			false_negative_count[true_class] += 1
			false_positive_count[pred_class] += 1

	sum_tp = true_positive_count.sum()
	sum_fp = false_positive_count.sum()
	sum_fn = false_negative_count.sum()
	precision = sum_tp/(sum_tp + sum_fp)
	recall = sum_tp/(sum_tp + sum_fn)
	if(precision == 0 or recall == 0):
		# print(precision, recall)
		precision = 1
		recall = 1

	mean_f_score = 2.0/((1.0/precision)+(1.0/recall))

	return ((num_samples-num_correct) * 1.0/num_samples, loss * 1.0/num_samples, mean_f_score)

def init_theta():
	theta = []
	theta.append(W_layer)
	theta.append(B_layer)
	return theta

def init_d_theta():
	d_theta = [[], []]
	for i in range(2):
		d_theta[i].append(-1)
	for j in xrange(1, L+1):
		d_theta[0].append(np.zeros(W_layer[j].shape))
		d_theta[1].append(np.zeros(B_layer[j].shape))
	return d_theta

def update_adam_factors(d_theta, d_theta_sq):
	for i in range(2):
		for j in xrange(len(d_theta[i])):
			d_theta_sq[i][j] = np.square(d_theta[i][j])

def add_and_set_theta(theta1, theta2):
	for i in range(2):
		for j in xrange(len(theta1[i])):
			theta1[i][j] += theta2[i][j]

def sub_and_set_theta(theta1, theta2):
	for i in range(2):
		for j in xrange(len(theta1[i])):
			theta1[i][j] -= theta2[i][j]

def scalar_mul_theta(theta, a):
	for i in range(2):
		for j in xrange(len(theta[i])):
			theta[i][j] *= a;

def copy_to_theta(theta, theta1):
	for i in range(2):
		for j in xrange(len(theta[i])):
			theta[i][j] = theta1[i][j]

def adam_decay_scale(m_t, v_t, epsilon):
	for i in range(2):
		for j in xrange(1, len(m_t[i])):
			m_t[i][j] *= ( 1.0 / np.sqrt( epsilon + v_t[i][j] ) )

def save_theta(theta, theta_pickle_file):
	pickle.dump(theta, theta_pickle_file)
	theta_pickle_file.close()
	# TODO

def do_mini_batch_gradient_descent(X, Y):
	d_theta = init_d_theta()
	prev_epoch_val_loss = float('inf')
	prev_epoch_train_loss = float('inf')
	prev_epoch_theta = []
	val_loss = 1
	train_loss = 1
	i = 0
	global lr
	while i < max_epochs:
		X,Y = sk.shuffle(X,Y)
		num_points_seen = 0
		steps = 0
		for x,y in zip(X,Y):
			y_hat = forward_propagation(x)
			add_and_set_theta(d_theta, backward_propagation(y, y_hat))
			num_points_seen += 1
	  		if(num_points_seen % batch_size == 0):
	  			# seen one mini batch
	  			scalar_mul_theta(d_theta, lr)
	  			sub_and_set_theta(theta, d_theta)
	  			steps += 1
	  			scalar_mul_theta(d_theta, 0)
		  		if steps % log_frequency == 0:
		  			train_error, train_loss, train_score = calc_error_loss(X, Y)
		  			val_error, val_loss, val_score = calc_error_loss(X_val, Y_val)
		  			log_train_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, train_loss, train_error, lr, train_score))
		  			log_val_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, val_loss, val_error, lr, val_score))
		pickle.dump(theta, theta_pickle_file)

		repeat_epoch = False
		if anneal_type == "val":
			repeat_epoch = (val_loss > prev_epoch_val_loss)
		else:
			repeat_epoch = (train_loss > prev_epoch_train_loss)
		
		if (anneal == True) and repeat_epoch:
			copy_to_theta(theta, prev_epoch_theta)
			if (lr < anneal_threshold):
				return;
			lr /= 2
		else:
			if (anneal == True):
				prev_epoch_theta = copy.deepcopy(theta)
				prev_epoch_train_loss = train_loss
				prev_epoch_val_loss = val_loss
			i += 1

def momentum_gradient_descent(X, Y):
	d_theta = init_d_theta()
	update = init_d_theta()

	prev_epoch_val_loss = float('inf')
	prev_epoch_train_loss = float('inf')
	prev_epoch_theta = []
	val_loss = 1
	train_loss = 1
	i = 0
	global lr
	while i < max_epochs:
		X,Y = sk.shuffle(X,Y)
		num_points_seen = 0
		steps = 0
		for x,y in zip(X,Y):
			y_hat = forward_propagation(x)
			add_and_set_theta(d_theta, backward_propagation(y, y_hat))
			num_points_seen += 1
	  		if(num_points_seen % batch_size == 0):
	  			# seen one mini batch
	  			scalar_mul_theta(update, momentum)
	  			scalar_mul_theta(d_theta, lr)
	  			add_and_set_theta(update, d_theta)
	  			sub_and_set_theta(theta, update)
	  			steps += 1
	  			scalar_mul_theta(d_theta, 0)
		  		if steps % log_frequency == 0:
		  			train_error, train_loss, train_score = calc_error_loss(X, Y)
		  			val_error, val_loss, val_score = calc_error_loss(X_val, Y_val)
		  			log_train_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, train_loss, train_error, lr, train_score))
		  			log_val_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, val_loss, val_error, lr, val_score))
		pickle.dump(theta, theta_pickle_file)

		repeat_epoch = False
		if anneal_type == "val":
			repeat_epoch = (val_loss > prev_epoch_val_loss)
		else:
			repeat_epoch = (train_loss > prev_epoch_train_loss)
		
		if (anneal == True) and repeat_epoch:
			copy_to_theta(theta, prev_epoch_theta)
			if (lr < anneal_threshold):
				return;
			lr /= 2
		else:
			if (anneal == True):
				prev_epoch_theta = copy.deepcopy(theta)
				prev_epoch_train_loss = train_loss
				prev_epoch_val_loss = val_loss
			i += 1

def nag_gradient_descent(X, Y):
	d_theta = init_d_theta()
	update = init_d_theta()

	prev_epoch_val_loss = float('inf')
	prev_epoch_train_loss = float('inf')
	prev_epoch_theta = []
	val_loss = 1
	train_loss = 1
	i = 0
	global lr
	while i < max_epochs:
		X,Y = sk.shuffle(X,Y)
		num_points_seen = 0
		steps = 0
		for x,y in zip(X,Y):
			if(num_points_seen % batch_size == 0):
				scalar_mul_theta(update, momentum)
				sub_and_set_theta(theta, update)				
			y_hat = forward_propagation(x)
			add_and_set_theta(d_theta, backward_propagation(y, y_hat))
			num_points_seen += 1
	  		if(num_points_seen % batch_size == 0):
	  			# seen one mini batch
	  			scalar_mul_theta(d_theta, lr)
	  			add_and_set_theta(update, d_theta)
	  			sub_and_set_theta(theta, d_theta)
	  			steps += 1
	  			scalar_mul_theta(d_theta, 0)
		  		if steps % log_frequency == 0:
		  			train_error, train_loss, train_score = calc_error_loss(X, Y)
		  			val_error, val_loss, val_score = calc_error_loss(X_val, Y_val)
		  			log_train_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, train_loss, train_error, lr, train_score))
		  			log_val_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps, val_loss, val_error, lr, val_score))
		pickle.dump(theta, theta_pickle_file)

		repeat_epoch = False
		if anneal_type == "val":
			repeat_epoch = (val_loss > prev_epoch_val_loss)
		else:
			repeat_epoch = (train_loss > prev_epoch_train_loss)
		
		if (anneal == True) and repeat_epoch:
			copy_to_theta(theta, prev_epoch_theta)
			if (lr < anneal_threshold):
				return;
			lr /= 2
		else:
			if (anneal == True):
				prev_epoch_theta = copy.deepcopy(theta)
				prev_epoch_train_loss = train_loss
				prev_epoch_val_loss = val_loss
			i += 1

def adam_gradient_descent(X, Y):
	beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
	d_theta = init_d_theta()
	m_t = init_d_theta()
	v_t = init_d_theta()
	d_theta_sq = init_d_theta()

	prev_epoch_val_loss = float('inf')
	prev_epoch_train_loss = float('inf')
	prev_epoch_theta = []
	prev_m_t = []
	prev_v_t = []
	val_loss = 1
	train_loss = 1
	i = 0
	global lr
	steps = 0
	prev_steps = 0
	while i < max_epochs:
		steps_print = 0
		X,Y = sk.shuffle(X,Y)
		num_points_seen = 0
		for x,y in zip(X,Y):
			masks = [-1]
			y_hat = forward_propagation_with_dropouts(x, masks)
			add_and_set_theta(d_theta, backward_propagation_with_dropouts(y, y_hat, masks))
			num_points_seen += 1
	  		if(num_points_seen % batch_size == 0):
	  			# seen one mini batch
	  			scalar_mul_theta(v_t, beta_2)
	  			update_adam_factors(d_theta, d_theta_sq)
	  			scalar_mul_theta(d_theta_sq, 1 - beta_2)
	  			add_and_set_theta(v_t, d_theta_sq)
	  			
	  			scalar_mul_theta(m_t, beta_1)
	  			scalar_mul_theta(d_theta, 1 - beta_1)
	  			add_and_set_theta(m_t, d_theta)

	  			steps += 1
	  			steps_print += 1
	  			temp_m_t = copy.deepcopy(m_t)
	  			temp_v_t = copy.deepcopy(v_t)

	  			scalar_mul_theta(temp_m_t, (1.0 / (1.0 - np.power(beta_1, steps))))
	  			scalar_mul_theta(temp_v_t, (1.0 / (1.0 - np.power(beta_2, steps))))

	  			adam_decay_scale(temp_m_t, temp_v_t, epsilon)
	  			scalar_mul_theta(temp_m_t, lr)
	  			sub_and_set_theta(theta, temp_m_t)
	  			
	  			scalar_mul_theta(d_theta, 0)
		  		if steps % log_frequency == 0:
		  			train_error, train_loss, train_score = calc_error_loss(X, Y)
		  			val_error, val_loss, val_score = calc_error_loss(X_val, Y_val)
		  			log_train_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps_print, train_loss, train_error, lr, train_score))
		  			log_val_file.write("Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}, score: {}\n".format(i, steps_print, val_loss, val_error, lr, val_score))
		pickle.dump(theta, theta_pickle_file)

		repeat_epoch = False
		if anneal_type == "val":
			repeat_epoch = (val_loss > prev_epoch_val_loss)
		else:
			repeat_epoch = (train_loss > prev_epoch_train_loss)
		
		if (anneal == True) and repeat_epoch:
			copy_to_theta(theta, prev_epoch_theta)
			copy_to_theta(m_t, prev_m_t)
			copy_to_theta(v_t, prev_v_t)
			steps = prev_steps
			if (lr < anneal_threshold):
				return;
			lr /= 2
		else:
			if (anneal == True):
				prev_epoch_theta = copy.deepcopy(theta)
				prev_m_t = copy.deepcopy(m_t)
				prev_v_t = copy.deepcopy(v_t)
				prev_epoch_train_loss = train_loss
				prev_epoch_val_loss = val_loss
				prev_steps = steps
			i += 1


def forward_propagation(x):
	H_layer[0] = x
	for i in xrange(1, L):
		A_layer[i] = B_layer[i] + np.matmul(W_layer[i], H_layer[i-1])
		H_layer[i] = activation_func(A_layer[i])
	A_layer[L] = B_layer[L] + np.matmul(W_layer[L], H_layer[L-1])
	y_hat = softmax(A_layer[L]) 
	return y_hat

def forward_propagation_with_dropouts(x, masks):
	p = 0.85	
	H_layer[0] = x
	for i in xrange(1, L):
		A_layer[i] = B_layer[i] + np.matmul(W_layer[i], H_layer[i-1])
		H_layer[i] = activation_func(A_layer[i])
		U = (np.random.rand(*H_layer[i].shape) < p) / p
		H_layer[i] *= U
		masks.append(U)
	A_layer[L] = B_layer[L] + np.matmul(W_layer[L], H_layer[L-1])
	y_hat = softmax(A_layer[L]) 
	return y_hat

# TODO: find gradients for squared error - Done. Verify.
def backward_propagation(y, y_hat):
	d_H = [-1]
	d_A = [-1]
	d_B = [-1]
	d_W = [-1]
	for i in xrange(1, L):
		d_H.append(np.zeros(H_layer[i].shape))
	for i in xrange(1, L+1):
		d_A.append(np.zeros(A_layer[i].shape))
	for i in xrange(1, L+1):
		d_B.append(np.zeros(B_layer[i].shape))
	for i in xrange(1, L+1):
		d_W.append(np.zeros(W_layer[i].shape))
	
	# Output gradient computation
	if(loss == 'ce'):
		d_A[L] = -(y - y_hat)
	else:
		temp_1 = 2 * y_hat
		temp_2 = y_hat - y + ( ( y_hat[np.argmax(y)] - np.square(y_hat).sum() ) * np.ones([10,1]) )
		d_A[L] = temp_1 * temp_2

	for k in xrange(L, 0, -1):
		# Parameter gradient computation
		d_W[k] = np.matmul(d_A[k], H_layer[k-1].transpose())
		d_B[k] = d_A[k]

		# Means we have already computed till W[1] and B[1]
		if(k == 1):
			break

		# Compute gradient wrt layer below
		d_H[k-1] = np.matmul(W_layer[k].transpose(),d_A[k])

		# Compute gradient wrt layer below (pre-activation)
		temp = activation_der(A_layer[k-1])
		d_A[k-1] = d_H[k-1] * temp

	d_theta = []
	d_theta.append(d_W)
	d_theta.append(d_B)
	scalar_mul_theta(d_theta, 1.0/batch_size)
	return d_theta

def backward_propagation_with_dropouts(y, y_hat, masks):
	d_H = [-1]
	d_A = [-1]
	d_B = [-1]
	d_W = [-1]
	for i in xrange(1, L):
		d_H.append(np.zeros(H_layer[i].shape))
	for i in xrange(1, L+1):
		d_A.append(np.zeros(A_layer[i].shape))
	for i in xrange(1, L+1):
		d_B.append(np.zeros(B_layer[i].shape))
	for i in xrange(1, L+1):
		d_W.append(np.zeros(W_layer[i].shape))
	
	# Output gradient computation
	if(loss == 'ce'):
		d_A[L] = -(y - y_hat)
	else:
		temp_1 = 2 * y_hat
		temp_2 = y_hat - y + ( ( y_hat[np.argmax(y)] - np.square(y_hat).sum() ) * np.ones([10,1]) )
		d_A[L] = temp_1 * temp_2

	for k in xrange(L, 0, -1):
		# Parameter gradient computation
		d_W[k] = np.matmul(d_A[k], H_layer[k-1].transpose())
		d_B[k] = d_A[k]

		# Means we have already computed till W[1] and B[1]
		if(k == 1):
			break

		# Compute gradient wrt layer below
		d_H[k-1] = np.matmul(W_layer[k].transpose(),d_A[k]) * masks[k-1] 

		# Compute gradient wrt layer below (pre-activation)
		temp = activation_der(A_layer[k-1])
		d_A[k-1] = d_H[k-1] * temp

	d_theta = []
	d_theta.append(d_W)
	d_theta.append(d_B)
	scalar_mul_theta(d_theta, 1.0/batch_size)
	return d_theta

if __name__ == '__main__':
	np.random.seed(1234)
	print_params()
	main()
