#!/bin/bash
#!/bin/bash
# set -x

data_dir=Data/plot_data

make_csv() {
	if [ ! $# -eq 1 ]; then return; fi
	parent_dir=$(dirname "$1")
	csv_file=$parent_dir/$(basename "$1" .txt).csv
	echo "idx,epoch,step,loss,error,lr,score" > $csv_file
	sed -rn 's/(.*): Epoch (.*), Step (.*), Loss: (.*), Error: (.*), lr: (.*), score: (.*)/\1,\2,\3,\4,\5,\6,\7/p' $1 >> $csv_file
}



# plots 1-4
for num_hidden in 1 2 3 4; do
	dir_name=plot_$num_hidden
	# subplots
	for hidden_layer_size in 50 100 200 300; do
		cdir=$data_dir/$dir_name/layer_size_$hidden_layer_size
		make_csv $cdir/expt/log_train.txt
		make_csv $cdir/expt/log_val.txt
	done
done

# plot 5
dir_name=plot_5
for opt in adam nag momentum gd; do
	cdir=$data_dir/$dir_name/opt_$opt
	make_csv $cdir/expt/log_train.txt
	make_csv $cdir/expt/log_val.txt
done

# plot 6
dir_name=plot_6
for activation in sigmoid tanh; do
	cdir=$data_dir/$dir_name/activation_$activation
	make_csv $cdir/expt/log_train.txt
	make_csv $cdir/expt/log_val.txt
done

# plot 7
dir_name=plot_7
for loss in sq ce; do
	cdir=$data_dir/$dir_name/loss_$loss
	make_csv $cdir/expt/log_train.txt
	make_csv $cdir/expt/log_val.txt
done

# plot 8
dir_name=plot_8
for batch_size in 1 20 100 1000; do
	cdir=$data_dir/$dir_name/batch_size_$batch_size
	make_csv $cdir/expt/log_train.txt
	make_csv $cdir/expt/log_val.txt
done



