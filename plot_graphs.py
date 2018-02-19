import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ipdb

data_dir = "Data/plot_data"
plot_dir = "Data/plots_2"

max_epochs = 20
x_epochs = range(1,max_epochs+1)
layer_sizes = [50, 100, 200, 300]

# plots 1-4
for num_hidden in [1, 2, 3, 4]:
	dir_name = str("plot_"+str(num_hidden))
	# subplots
	y_train_loss_list = []
	y_val_loss_list = []
	for hidden_layer_size in layer_sizes:
		cdir = os.path.join(data_dir, dir_name, "layer_size_"+str(hidden_layer_size))
		train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
		train_loss = train_df['loss'].tolist()
		y_train_loss_list.append(train_loss)

		val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
		val_loss = val_df['loss'].tolist()
		y_val_loss_list.append(val_loss)
	plt.clf()
	plt.subplot(2,1,1)
	for i in range(4):
		plt.plot(x_epochs, y_train_loss_list[i], label=str(layer_sizes[i]), alpha=0.7)
	plt.title("{} hidden layer(s)".format(num_hidden))
	plt.xlabel("epochs")
	plt.xticks(x_epochs)
	plt.ylabel("training loss")
	plt.gca().set_ylim([0,1])
	# plt.axis('scaled')
	plt.gca().set_aspect('equal', 'datalim')
	# plt.gca().set_aspect('equal', 'box')
	plt.legend(loc="best", title="layer size")
	plt.grid()

	plt.subplot(2,1,2)
	for i in range(4):
		plt.plot(x_epochs, y_val_loss_list[i], label=str(layer_sizes[i]), alpha=0.7)
	plt.xlabel("epochs")
	plt.xticks(x_epochs)
	plt.ylabel("validation loss")
	plt.gca().set_ylim([0,1])
	# plt.axis('scaled')
	plt.gca().set_aspect('equal', 'datalim')
	# plt.gca().set_aspect('equal', 'box')
	plt.legend(loc="best", title="layer size")
	plt.grid()

	plt.tight_layout()
	plt.savefig(os.path.join(plot_dir, "plot_{}.png".format(num_hidden)))


# plot 5
dir_name = "plot_5"
opts = ["adam", "nag", "momentum", "gd"]
y_train_loss_list = []
y_val_loss_list = []

for opt in opts:
	cdir = os.path.join(data_dir, dir_name, "opt_"+opt)
	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
	train_loss = train_df['loss'].tolist()[1:]
	y_train_loss_list.append(train_loss)

	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
	val_loss = val_df['loss'].tolist()[1:]
	y_val_loss_list.append(val_loss)
	
plt.clf()
plt.subplot(2,1,1)
for i in range(4):
	plt.plot(x_epochs[1:], y_train_loss_list[i], label=opts[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("training loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="opt")
plt.grid()

plt.subplot(2,1,2)
for i in range(4):
	plt.plot(x_epochs[1:], y_val_loss_list[i], label=opts[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("validation loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="opt")
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "plot_{}.png".format(5)))

# plot 6
y_train_loss_list = []
y_val_loss_list = []
dir_name = "plot_6"
activations = ["sigmoid", "tanh"]
for activation in activations:
	cdir = os.path.join(data_dir, dir_name, "activation_"+activation)
	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
	train_loss = train_df['loss'].tolist()
	y_train_loss_list.append(train_loss)

	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
	val_loss = val_df['loss'].tolist()
	y_val_loss_list.append(val_loss)

plt.clf()
plt.subplot(2,1,1)
for i in range(2):
	plt.plot(x_epochs, y_train_loss_list[i], label=activations[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("training loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="activation")
plt.grid()

plt.subplot(2,1,2)
for i in range(2):
	plt.plot(x_epochs, y_val_loss_list[i], label=activations[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("validation loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="activation")
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "plot_{}.png".format(6)))

# plot 7
y_train_loss_list = []
y_val_loss_list = []
dir_name = "plot_7"
losses = ["sq", "ce"]
for loss in losses:
	cdir = os.path.join(data_dir, dir_name, "loss_"+loss)
	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
	train_loss = train_df['loss'].tolist()
	y_train_loss_list.append(train_loss)

	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
	val_loss = val_df['loss'].tolist()
	y_val_loss_list.append(val_loss)

plt.clf()
plt.subplot(2,1,1)
for i in range(2):
	plt.plot(x_epochs, y_train_loss_list[i], label=losses[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("training loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="loss")
plt.grid()

plt.subplot(2,1,2)
for i in range(2):
	plt.plot(x_epochs, y_val_loss_list[i], label=losses[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("validation loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="loss")
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "plot_{}.png".format(7)))

# plot 8
y_train_loss_list = []
y_val_loss_list = []
dir_name = "plot_8"
batch_sizes = [1, 20, 100, 1000]
for batch_size in batch_sizes:
	cdir = os.path.join(data_dir, dir_name, "batch_size_"+str(batch_size))
	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
	train_loss = train_df['loss'].tolist()[1:]
	y_train_loss_list.append(train_loss)

	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
	val_loss = val_df['loss'].tolist()[1:]
	y_val_loss_list.append(val_loss)



plt.clf()
plt.subplot(2,1,1)
for i in range(4):
	plt.plot(x_epochs[1:], y_train_loss_list[i], label=batch_sizes[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("training loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="batch size")
plt.grid()

plt.subplot(2,1,2)
for i in range(4):
	plt.plot(x_epochs[1:], y_val_loss_list[i], label=batch_sizes[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("validation loss")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="batch size")
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "plot_{}.png".format(8)))

# plot cross entropy, squared loss (error vs epochs)
y_train_error_list = []
y_val_error_list = []
dir_name = "plot_7"
losses = ["sq", "ce"]
for loss in losses:
	cdir = os.path.join(data_dir, dir_name, "loss_"+loss)
	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
	train_error = train_df['error'].tolist()[1:]
	y_train_error_list.append(train_error)

	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
	val_error = val_df['error'].tolist()[1:]
	y_val_error_list.append(val_error)

plt.clf()
plt.subplot(2,1,1)
for i in range(2):
	plt.plot(x_epochs[1:], y_train_error_list[i], label=losses[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("training error")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="loss")
plt.grid()

plt.subplot(2,1,2)
for i in range(2):
	plt.plot(x_epochs[1:], y_val_error_list[i], label=losses[i], alpha=0.7)
plt.xlabel("epochs")
plt.xticks(x_epochs)
plt.ylabel("validation error")
plt.gca().set_ylim([0,1])
plt.legend(loc="best", title="loss")
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "plot_{}.png".format("7-2")))

"""
# plot batch sizes (with uniform initialization?)
# y_train_loss_list = []
# y_val_loss_list = []
# dir_name = "plot_8"
# batch_sizes = [1, 20, 100, 1000]
# for batch_size in batch_sizes:
# 	cdir = os.path.join(data_dir, dir_name, "batch_size_"+str(batch_size))
# 	train_df = pd.read_csv(os.path.join(cdir, "expt/log_train.csv"))
# 	train_loss = train_df['loss'].tolist()
# 	y_train_loss_list.append(train_loss)

# 	val_df = pd.read_csv(os.path.join(cdir, "expt/log_val.csv"))
# 	val_loss = val_df['loss'].tolist()
# 	y_val_loss_list.append(val_loss)

# plt.clf()
# plt.subplot(2,1,1)
# for i in range(4):
# 	plt.plot(x_epochs, y_train_loss_list[i], label=batch_sizes[i], alpha=0.7)
# plt.xlabel("epochs")
# plt.xticks(x_epochs)
# plt.ylabel("training loss")
# plt.gca().set_ylim([0,1])
# plt.legend(loc="best", title="batch size")
# plt.grid()

# plt.subplot(2,1,2)
# for i in range(4):
# 	plt.plot(x_epochs, y_val_loss_list[i], label=batch_sizes[i], alpha=0.7)
# plt.xlabel("epochs")
# plt.xticks(x_epochs)
# plt.ylabel("validation loss")
# plt.gca().set_ylim([0,1])
# plt.legend(loc="best", title="batch size")
# plt.grid()

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "plot_{}.png".format("8-2")))
"""