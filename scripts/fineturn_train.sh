#!/bin/bash
WORKSPACE=${1:-"/root/DCASE2023_data/correct_data_experiment/all_no_eval_data"}   # Default argument.
PRETRAINED_CHECKPOINT_PATH="/root/DCASE2023_data/code/train/cnn14_model/ResNet38_mAP=0.434.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_fineturn_e.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=800 \
    --hop_size=320 \
    --mel_bins=256 \
    --fmin=0 \
    --fmax=16000 \
    --model_type='Transfer_ResNet38_new_f' \
    --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH \
    --loss_type='nn_ce' \
    --balanced='alternate' \
    --augmentation='mixup' \
    --batch_size=100 \
    --learning_rate=1e-5 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda \
    --mixup_alpha=0.3 \
    --mixstyle_p=0.6 \
    --mixstyle_alpha=0.3 \
    --data_load_way='conv_ir' \
    --lr_strategy='none'

