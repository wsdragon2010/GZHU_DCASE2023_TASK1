#!/bin/bash
WORKSPACE=${1:-"/root/DCASE2023_data/correct_data_experiment/in_meta_not_in_eval_add_combine_device_20461"}   # Default argument.
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_dcase.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=2048 \
    --hop_size=1024 \
    --mel_bins=256 \
    --fmin=0 \
    --fmax=16000 \
    --model_type='BcRes2NetModel_resnorm_bn_quant' \
    --loss_type='nn_ce' \
    --balanced='alternate' \
    --augmentation='mixup' \
    --batch_size=100 \
    --learning_rate=1e-4 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda \
    --mixup_alpha=0.3 \
    --model_width=3 \
    --mixstyle_p=0.6 \
    --mixstyle_alpha=0.3 \

