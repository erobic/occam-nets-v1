#!/bin/bash
source activate occamnets

GPU=0
dataset=biased_mnist
optim=biased_mnist

for p in 0.95; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=resnet18 \
  trainer=pgi_trainer \
  trainer.invariance_loss_wt_coeff=100 \
  dataset=${dataset} \
  dataset.p_bias=${p} \
  optimizer=${optim} \
  expt_suffix=p_${p}

  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18 \
  trainer=occam_pgi_trainer \
  trainer.invariance_loss_wt_coeff=50 \
  dataset=${dataset} \
  dataset.p_bias=${p} \
  optimizer=${optim} \
  expt_suffix=p_${p}
done