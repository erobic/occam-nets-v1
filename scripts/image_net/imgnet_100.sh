#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net_100
optim=image_net
precision=16

#
#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=occam_resnet18 \
#trainer=occam_trainer_image_net \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.batch_size=128 \
#optimizer=${optim} \
#trainer.exit_gating.train_acc_thresholds=[0.2,0.3,0.4,0.5] \
#expt_suffix=prec_${precision}_thresh_0.2

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.batch_size=128 \
optimizer=${optim} \
expt_suffix=test_prec_${precision}_thresh_0.2 \
'checkpoint_path="/home/robik/occam_net_v1/image_net_100/OccamTrainer/occam_resnet18/prec_16_thresh_0.2/lightning_logs/version_0/checkpoints/epoch=89-step=90990.ckpt"' \
task.name='test' \
data_sub_split='val'


# _tmp \
  #trainer.limit_train_batches=2 \
  #trainer.limit_val_batches=2 \
  #trainer.limit_test_batches=2 \
  #optimizer.epochs=2