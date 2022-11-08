# absolute path that contains all datasets
DATA_ROOT=/media/bjw/disk1/depth_data

# # kitti
# DATASET=$DATA_ROOT/kitti
# CONFIG=configs/v1/kitti_raw.txt
# CKPT=ckpts/kitti_scv1/epoch=99-val_loss=0.1411.ckpt

# # nyu
# DATASET=$DATA_ROOT/nyu
# CONFIG=configs/v2/nyu.txt
# CKPT=ckpts/nyu_scv2/epoch=101-val_loss=0.1580.ckpt

# ddad
DATASET=$DATA_ROOT/ddad
CONFIG=configs/v3/ddad.txt
CKPT=ckpts/ddad_scv3/epoch=99-val_loss=0.1438.ckpt

# run
export CUDA_VISIBLE_DEVICES=0
python test.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT

