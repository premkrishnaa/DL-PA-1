#!/bin/bash
set -x

data_dir=Data
train_file=$data_dir/train.csv
test_file=$data_dir/test.csv
val_file=$data_dir/val.csv
momentum=0.5
activation=sigmoid
loss=ce
opt=adam
batch_size=20
lr=0.002

# plots 1-4
# for num_hidden in 1 2 3 4; do
# 	dir_name=plot_$num_hidden
# 	# subplots
# 	for hidden_layer_size in 50 100 200 300; do
# 		sizes=$(for ((i=0; i<$num_hidden-1; i++)); do printf $hidden_layer_size,; done; printf $hidden_layer_size)
# 		cdir=$data_dir/$dir_name/layer_size_$hidden_layer_size
# 		python -u train.py --lr $lr --momentum $momentum --num_hidden $num_hidden --sizes $sizes --activation $activation --loss $loss --opt $opt --batch_size $batch_size --anneal false --save_dir $cdir/pickle/ --expt_dir $cdir/expt/ --train $train_file --test $test_file --val $val_file --no-save_all_thetas
# 		sleep 5
# 	done
# done

# plot 5
num_hidden=3
hidden_layer_size=300
sizes=$(for ((i=0; i<$num_hidden-1; i++)); do printf $hidden_layer_size,; done; printf $hidden_layer_size)
dir_name=plot_5
for opt in adam nag momentum gd; do
	cdir=$data_dir/$dir_name/opt_$opt
	python -u train.py --lr $lr --momentum $momentum --num_hidden $num_hidden --sizes $sizes --activation $activation --loss $loss --opt $opt --batch_size $batch_size --anneal false --save_dir $cdir/pickle/ --expt_dir $cdir/expt/ --train $train_file --test $test_file --val $val_file --no-save_all_thetas
	# sleep 5
done

# plot 6
# num_hidden=2
# hidden_layer_size=100
# sizes=$(for ((i=0; i<$num_hidden-1; i++)); do printf $hidden_layer_size,; done; printf $hidden_layer_size)
# opt=adam
# dir_name=plot_6
# for activation in sigmoid tanh; do
# 	cdir=$data_dir/$dir_name/activation_$activation
# 	python -u train.py --lr $lr --momentum $momentum --num_hidden $num_hidden --sizes $sizes --activation $activation --loss $loss --opt $opt --batch_size $batch_size --anneal false --save_dir $cdir/pickle/ --expt_dir $cdir/expt/ --train $train_file --test $test_file --val $val_file --no-save_all_thetas
# 	sleep 5
# done

# # plot 7
# activation=sigmoid
# dir_name=plot_7
# for loss in sq ce; do
# 	cdir=$data_dir/$dir_name/loss_$loss
# 	python -u train.py --lr $lr --momentum $momentum --num_hidden $num_hidden --sizes $sizes --activation $activation --loss $loss --opt $opt --batch_size $batch_size --anneal false --save_dir $cdir/pickle/ --expt_dir $cdir/expt/ --train $train_file --test $test_file --val $val_file --no-save_all_thetas
# 	sleep 5
# done

# plot 8
loss=ce
dir_name=plot_8
for batch_size in 1000; do
	cdir=$data_dir/$dir_name/batch_size_$batch_size
	python -u train.py --lr $lr --momentum $momentum --num_hidden $num_hidden --sizes $sizes --activation $activation --loss $loss --opt $opt --batch_size $batch_size --anneal false --save_dir $cdir/pickle/ --expt_dir $cdir/expt/ --train $train_file --test $test_file --val $val_file --no-save_all_thetas
	# sleep 5
done



