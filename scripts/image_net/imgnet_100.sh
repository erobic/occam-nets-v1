#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net_100
optim=image_net
precision=16

#
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.batch_size=128 \
optimizer=${optim} \
trainer.exit_gating.train_acc_thresholds=[0.25,0.35,0.45,0.55] \
expt_suffix=prec_${precision}_thresh_0.25


# _tmp \
  #trainer.limit_train_batches=2 \
  #trainer.limit_val_batches=2 \
  #trainer.limit_test_batches=2 \
  #optimizer.epochs=2