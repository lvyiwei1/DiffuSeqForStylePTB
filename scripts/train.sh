#!/bin/bash

#SBATCH --job-name=diffuseq
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:4
#SBATCH --time=240:00:00
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mail-user=yiweilyu@umich.edu
#SBATCH --export=ALL


python -m torch.distributed.launch --nproc_per_node=4 --master_port=12234 --use_env run_train.py \
--diff_steps 1000 \
--lr 0.0001 \
--learning_steps 10000 \
--save_interval 1000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 1024 \
--dataset ATP \
--data_dir ~/DiffuSeq/datasets/ATP \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes ATP
