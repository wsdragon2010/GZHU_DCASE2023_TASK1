#!/bin/bash
WORKSPACE=${1:-"/root/DCASE2023_data/correct_data_experiment/in_meta_not_in_eval_add_combine_device_20461"}   # Default argument.
cnn14_path='/root/DCASE2023_data/code/train/cnn14_model/ResNet38_mAP=0.434.pth'
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_dcase_dml.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --pretrained_checkpoint_path=$cnn14_path \
    --window_size=2048 \
    --hop_size=1024 \
    --mel_bins=256 \
    --tea_window_size=800 \
    --tea_hop_size=320 \
    --tea_mel_bins=256 \
    --fmin=0 \
    --fmax=16000 \
    --model1_type='BcRes2NetModel_resnorm_bn_quant' \
    --model2_type='BcRes2NetModel_deep_quant' \
    --model3_type='BcRes2NetModel_width_resnorm_bn_quant' \
    --model4_type='Transfer_ResNet38_new_f' \
    --loss_type='nn_ce' \
    --balanced='alternate' \
    --augmentation='mixstyle' \
    --batch_size=100 \
    --learning_rate=1e-4 \
    --tea_learning_rate=1e-4 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda \
    --mixup_alpha=0.3 \
    --mixstyle_p=0.6 \
    --mixstyle_alpha=0.3 \
    --small_model_width=3 \
    --large_model_width=10 \
    --T=3 \
    --data_load_way='conv_ir' \
    --lr_strategy='none'

# Plot statistics
# python3 utils/plot_statistics.py plot \
#     --dataset_dir=$DATASET_DIR \
#     --workspace=$WORKSPACE \
#     --select=1_aug
