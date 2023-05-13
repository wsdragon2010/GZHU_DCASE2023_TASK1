#!/bin/bash

scp_path='/private/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/dcase_eval_device'
workspace='/private/dcase22_reassembled_test/TAU-urban-acoustic-scenes-2022-mobile-development/hdf5s/dcase_eval_with_device'

total_lines=$(wc -l < $scp_path.scp) # 计算原始文件的总行数
lines_per_part=$(( (total_lines + 20 - 1) / 20 )) # 计算每个文件应该有多少行
split -l $lines_per_part -d --additional-suffix=.scp $scp_path.scp $scp_path"_part" # 按照计算出来的行数拆分文件

# get waveform.h5
for IDX in $(seq 0 19)
do
    # echo $IDX   
    j=$(printf "%02d" $IDX) # 将i格式化为两位数，不足的用0补齐，并赋值给j
    echo $j
    python3 utils/dataset_scp.py pack_waveforms_to_hdf5_dcase \
        --csv_path=$scp_path"_part"$j.scp \
        --waveforms_hdf5_path=$workspace"/waveforms/train_part"$j.h5 & # &和wait搭配使用，会使得当前命令进入后台，可并行执行
done
wait

rm $scp_path"_part"*.scp

# get indexes.h5
for IDX in $(seq 0 19)
do
    # echo $IDX
    j=$(printf "%02d" $IDX) # 将i格式化为两位数，不足的用0补齐，并赋值给j
    echo $j
    python3 utils/create_indexes.py create_indexes_dcase \
        --waveforms_hdf5_path=$workspace"/waveforms/train_part$j.h5" \
        --indexes_hdf5_path=$workspace"/indexes/train_part$j.h5" &
done
wait

# Combine balanced and unbalanced training indexes to a full training indexes hdf5
python3 utils/create_indexes.py combine_full_indexes_dcase \
    --indexes_hdf5s_dir=$workspace"/indexes" \
    --full_indexes_hdf5_path=$workspace"/indexes/train_eval.h5"
