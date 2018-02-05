#!/bin/sh
# python 
# python train.py --lr 0.01 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt adam --batch_size 5 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train train_10.csv --test Data/test.csv --val val_20.csv
# python train.py --lr 0.01 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt adam --batch_size 20 --anneal true --save_dir Data/pickle/ --expt_dir Data/expt/ --train Data/train.csv --test Data/test.csv --val Data/val.csv
python train.py --lr 0.05 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt adam --batch_size 20 --anneal true --save_dir Data/pickle/ --expt_dir Data/expt/ --train Data/train.csv --test Data/test.csv --val Data/val.csv