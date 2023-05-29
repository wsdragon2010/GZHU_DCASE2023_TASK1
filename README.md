# GZHU_DCASE2023_TASK1
This code base was used for the GZHU team's submission to DCASE2023 Task I Low Complexity Acoustic Scene Classification. The technical report corresponding to it can be found in [DCASE2023](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification).\
The entire code framework was written based on [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) and the pre-trained model we used, resnet38, can be downloaded in [there](https://zenodo.org/record/3987831).This code can be run using the same environment as PANNs.
## Reassemble Audio
First, use the open source code of [CPJKU](https://github.com/CPJKU/cpjku_dcase22) to reassemble the 1s audio clip into 10s audio
## Create hdf5
First use a scp file with spaces as spacers to store the meta information, fill in the order of audio name, label, path, device label. Then modify the "scp_path" and "workspace" in h5.sh to the corresponding paths in your file system, where workspace is the path where h5 is stored.Then run\
`sh scripts/h5.sh`
## Training
After Hdf5 is created, you can enter the training phase, first create a workspace for your training logs and models and other information. And "full_train.h5" and "eval.h5" are stored in your "workspace/indexes", and in and change "WORKSPACE" to the path you created in "\*_train.sh".
