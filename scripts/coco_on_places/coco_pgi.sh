#!/bin/bash
source activate occamnets

GPU=0
dataset=coco_on_places
optim=coco_on_places

for inv_wt in 50; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=resnet18_img64 \
  trainer=pgi_trainer \
  trainer.invariance_loss_wt_coeff=${inv_wt} \
  dataset=${dataset} \
  optimizer=${optim} \
  expt_suffix=inv_${inv_wt}
done

for inv_wt in 1 10; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18_img64 \
  trainer=occam_pgi_trainer \
  trainer.invariance_loss_wt_coeff=${inv_wt} \
  dataset=${dataset} \
  optimizer=${optim} \
  expt_suffix=inv_${inv_wt}
done