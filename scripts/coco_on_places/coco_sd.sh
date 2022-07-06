#!/bin/bash
source activate occamnets

GPU=1
dataset=coco_on_places
optim=coco_on_places

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=resnet18_img64 \
trainer=spectral_decoupling_trainer \
trainer.group_by='maj_min_group_ix' \
trainer.lambdas=[0.1,1e-3] \
trainer.gammas=[0,0] \
dataset=${dataset} \
optimizer=${optim}

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_img64 \
trainer=occam_spectral_decoupling_trainer \
trainer.group_by='maj_min_group_ix' \
trainer.lambdas=[0.1,1e-3] \
trainer.gammas=[0,0] \
dataset=${dataset} \
optimizer=${optim}