#!/bin/bash
WORKSPACE=${1:-"/private/correct_data_experiments/in_meta_not_in_eval_add_combine_device_20461"}   # Default argument.
teacher_model_path_with_1s='/private/model/dml_train_res38/a10_dml_bs100_e-4_mixstyle0.3_p0.dml_mixstyle_convir_Transfer_ResNet38_new_f_536000_iterations_acc7239_loss8369.pth'
stu_model_path='/private/model/dml_train_bcres2net/dml_mixstyle_conv_ir_BcRes2NetModel_448000_iterations_acc5815_loss1578.pth'
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_kd.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=2048 \
    --hop_size=1024 \
    --mel_bins=256 \
    --tea_window_size=800 \
    --tea_hop_size=320 \
    --tea_mel_bins=256 \
    --fmin=0 \
    --fmax=16000 \
    --model_type='BcRes2NetModel_resnorm_bn_quant' \
    --tea_model_type='Transfer_ResNet38_new_f' \
    --pretrained_checkpoint_path=$teacher_model_path_with_1s \
    --stu_pretrained_checkpoint_path=$stu_model_path \
    --loss_type='nn_ce' \
    --balanced='alternate' \
    --augmentation='mixstyle' \
    --batch_size=100 \
    --learning_rate=1e-4 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda \
    --model_width=3 \
    --mixup_alpha=0.3 \
    --mixstyle_p=0.6 \
    --mixstyle_alpha=0.3 \
    --T=3 \
    --kd_lambda=50 \
    --data='conv_ir' \
    --data_load_way='conv_ir' \
    --lr_strategy='none'

