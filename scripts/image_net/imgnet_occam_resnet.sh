#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.batch_size=256 \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_prec_${precision} \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8_prec_16/lightning_logs/version_0/checkpoints/epoch=39-step=16040.ckpt"' \
task.name='test' \
data_sub_split='val_mask'

#/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8_supp_0_prec_16/lightning_logs/version_1/checkpoints

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
trainer.cam_suppression.loss_wt=0 \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_supp_0_prec_${precision} \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8_supp_0_prec_16/lightning_logs/version_1/checkpoints/epoch=39-step=8040.ckpt"' \
task.name='test' \
data_sub_split='val_mask'

#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=resnet18 \
#trainer=base_trainer \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.subset_percent=${subset_percent} \
#optimizer=${optim} \
#expt_suffix=subset_${subset_percent}_prec_${precision}
#
#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=occam_resnet18 \
#trainer=occam_trainer_image_net \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.batch_size=256 \
#dataset.subset_percent=${subset_percent} \
#optimizer=${optim} \
#expt_suffix=subset_${subset_percent}_prec_${precision}
#
#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=occam_resnet18 \
#trainer=occam_trainer_image_net \
#trainer.precision=${precision} \
#trainer.cam_suppression.loss_wt=0 \
#dataset=${dataset} \
#dataset.subset_percent=${subset_percent} \
#optimizer=${optim} \
#expt_suffix=subset_${subset_percent}_supp_0_prec_${precision}
