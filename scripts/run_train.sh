# absolute path that contains all datasets
DATA_ROOT=/media/bjw/disk1/depth_data

# # kitti
# DATASET=$DATA_ROOT/kitti
# CONFIG=configs/v1/kitti_raw.txt

# # nyu
# DATASET=$DATA_ROOT/nyu
# CONFIG=configs/v2/nyu.txt

# ddad
DATASET=$DATA_ROOT/ddad
CONFIG=configs/v3/ddad.txt

# run
export CUDA_VISIBLE_DEVICES=0
python train.py --config $CONFIG --dataset_dir $DATASET