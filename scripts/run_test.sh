# absolute path that contains all datasets
DATA_ROOT=/media/bjw/Disk/release_depth_data

# # kitti
# DATASET=$DATA_ROOT/kitti
# CONFIG=configs/v1/kitti_raw.txt
# CKPT=ckpts/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt

# nyu
DATASET=$DATA_ROOT/nyu
CONFIG=configs/v2/nyu.txt
CKPT=ckpts/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt

# run
python test.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT

