#!/bin/bash
source activate occamnets

GPU=1
dataset=biased_mnist
optim=biased_mnist

for p in 0.95; do

  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18 \
  trainer=occam_trainer \
  dataset=${dataset} \
  dataset.p_bias=${p} \
  optimizer=${optim} \
  expt_suffix=p_${p} \
  'checkpoint_path="/home/robik/occam-networks-outputs/biased_mnist/OccamTrainer/occam_resnet18/p_0.95/lightning_logs/version_2/checkpoints/epoch=89-step=35190.ckpt"' \
  task.name='test'
done