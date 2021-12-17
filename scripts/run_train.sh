# absolute path that contains all datasets
DATA_ROOT=/media/bjw/Disk/release_depth_data

# # kitti
# DATASET=$DATA_ROOT/kitti
# CONFIG=configs/v1/kitti_raw.txt

# nyu
DATASET=$DATA_ROOT/nyu
CONFIG=configs/v2/nyu.txt

python train.py --config $CONFIG --dataset_dir $DATASET