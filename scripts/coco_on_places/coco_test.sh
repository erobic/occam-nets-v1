#!/bin/bash
source activate occamnets

GPU=1
dataset=coco_on_places
optim=coco_on_places

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_img64 \
trainer=occam_trainer \
dataset=${dataset} \
optimizer=${optim} \
'checkpoint_path="/home/robik/occam-networks-outputs/coco_on_places/OccamTrainer/occam_resnet18_img64/lightning_logs/version_0/checkpoints/epoch=149-step=16800.ckpt"' \
task.name='test' \
data_sub_split='sgtest'
