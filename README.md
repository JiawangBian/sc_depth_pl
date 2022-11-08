# SC_Depth:

This repo provides the pytorch lightning implementation of **SC-Depth** (V1, V2, and V3) for **self-supervised learning of monocular depth from video**.

In the SC-DepthV1 ([IJCV 2021](https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf) & [NeurIPS 2019](https://papers.nips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)), we propose (i) **geometry consistency loss** for scale-consistent depth prediction over time and (ii) **self-discovered mask** for detecting and removing dynamic regions and occlusions during training towards higher accuracy. The predicted depth is sufficiently accurate and consistent for use in the ORB-SLAM2 system. The below video showcases the estimated depth in the form of pointcloud (top) and color map (bottom right).

[<img src="https://jwbian.net/wp-content/uploads/2020/06/77CXZX@H37PIWDBX0R7T.png" width="400">](https://www.youtube.com/watch?v=OkfK3wmMnpo)

In the SC-DepthV2 ([TPMAI 2022](https://arxiv.org/abs/2006.02708v2)), we prove that the large relative rotational motions in the hand-held camera captured videos is the main challenge for unsupervised monocular depth estimation in indoor scenes. Based on this findings, we propose auto-recitify network (**ARN**) to handle the large relative rotation between consecutive video frames. It is integrated into SC-DepthV1 and jointly trained with self-supervised losses, greatly boosting the performance.

<img src="https://jwbian.net/wp-content/uploads/2020/06/vis_depth.png" width="400">

In the SC-DepthV3 ([ArXiv 2022](https://arxiv.org/abs/2211.03660)), we propose a robust learning framework for accurate and sharp monocular depth estimation in (highly) dynamic scenes. As the photometric loss, which is the main loss in the self-supervised methods, is not valid in dynamic object regions and occlusion, previous methods show poor accuracy in dynamic scenes and blurred depth prediction at object boundaries. We propose to leverage an external pretrained depth estimation network for generating the single-image depth prior, based on which we propose effective losses to constrain self-supervised depth learning. The evaluation results on six challenging datasets including both static and dynamic scenes demonstrate the efficacy of the proposed method.

Qualatative depth estimation results: DDAD, BONN, TUM, IBIMS-1

<img src="https://jwbian.net/Demo/vis_ddad.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_bonn.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_tum.jpg" width="400"> <img src="https://jwbian.net/Demo/vis_ibims.jpg" width="400">

Demo Videos

[<img src="https://img.youtube.com/vi/Mzd_csVpjys/hqdefault.jpg" width="400">](https://youtu.be/Mzd_csVpjys)
[<img src="https://img.youtube.com/vi/E-F2VVYVHFQ/hqdefault.jpg" width="400">](https://youtu.be/E-F2VVYVHFQ)

## Install
```
conda create -n sc_depth_env python=3.8
conda activate sc_depth_env
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

## Dataset

We organize the video datasets into the following format for training and testing models:

    Dataset
      -Training
        --Scene0000
          ---*.jpg (list of color images)
          ---cam.txt (3x3 camera intrinsic matrix)
          ---depth (a folder containing ground-truth depths, optional for validation)
        --Scene0001
        ...
        train.txt (containing training scene names)
        val.txt (containing validation scene names)
      -Testing
        --color (containg testing images)
        --depth (containg ground-truth depths)

We provide the pre-processed standard datasets:

[**[kitti_raw]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUax6F2N-rjAs43R?e=gwn6Zi) [**[nyu]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUUA5hElvhZXnqOn?e=51SIE1) [**more datasets are ongoing**]


## Training

We provide a bash script ("scripts/run_train.sh"), which shows how to train on kitti and nyu datasets. Generally, you need edit the config file (e.g., "configs/v1/kitti.txt") based on your devices and run
```bash
python train.py --config $CONFIG --dataset_dir $DATASET
```
Then you can start a `tensorboard` session in this folder by running
```bash
tensorboard --logdir=ckpts/
```
By opening [https://localhost:6006](https://localhost:6006) on your browser, you can watch the training progress.  


## Train on Your Own Data

You need re-organize your own video datasets according to the above mentioned format for training. Then, you may meet two problems: (1) no ground-truth depth for validation, and (2) hard to choose an appropriate frame rate (FPS) to downsample videos.

For (1), just add "--val_mode photo" in the training script or the configure file, which uses the photometric loss for validation. 
```bash
python train.py --config $CONFIG --dataset_dir $DATASET --val_mode photo
```

For (2), we provide a script ("generate_valid_frame_index.py"), which computes and saves a "frame_index.txt" in each training scene. You can call it by running
```bash
python generate_valid_frame_index.py --dataset_dir $DATASET
```
Then, you can add "--use_frame_index" in the training script or the configure file to train models on the filtered frames.
```bash
python train.py --config $CONFIG --dataset_dir $DATASET --use_frame_index
```


## Testing

We provide the script ("scripts/run_test.sh"), which shows how to test on kitti and nyu datasets.

    python test.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT


## Inference

We provide a bash script ("scripts/run_inference.sh"), which shows how to save the predicted depth (.npy) and visualize it using a color image (.jpg).
A demo is given here. You can put your images in "demo/input/" folder and run
```bash
python inference.py --config configs/v2/nyu.txt \
--input_dir demo/input/ \
--output_dir demo/output/ \
--ckpt_path ckpts/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt \
--save-vis --save-depth
```
You will see the results saved in "demo/output/" folder.


## Pretrained models

We provide pretrained models on kitti and nyu datasets. You need uncompress and put it into "ckpts" folder. Then you can run "scripts/run_test.sh" with the pretrained model (fill your real dataset path), and you will get the following results:

[**[kitti_scv1_model]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUNoHDuA2FKjjioD?e=fD8Ish):

|  Models  | Abs Rel | Sq Rel | Log10 | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|----------|---------|--------|-------|-------|-----------|-------|-------|-------|
| resnet18 | 0.119   | 0.878  | 0.053 | 4.987 | 0.196     | 0.859 | 0.956 | 0.981 |

 [**[nyu_scv2_model]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUSxFrPz690xaxwH?e=wFOR6A):

|  Models  | Abs Rel | Sq Rel | Log10 | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|----------|---------|--------|-------|-------|-----------|-------|-------|-------|
| resnet18 | 0.142   | 0.112  | 0.061 | 0.554 | 0.186     | 0.808 | 0.951 | 0.987 |

More pretrained models on more datasets will be provided!

## References

#### SC-DepthV1:
**Unsupervised Scale-consistent Depth Learning from Video (IJCV 2021)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Zhichao Li, Le Zhang, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf)
```
@article{bian2021ijcv, 
  title={Unsupervised Scale-consistent Depth Learning from Video}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Li, Zhichao and Zhang, Le and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian}, 
  journal= {International Journal of Computer Vision (IJCV)}, 
  year={2021} 
}
```
which is an extension of previous conference version:

**Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (NeurIPS 2019)** \
*Jia-Wang Bian, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://papers.nips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)
```
@inproceedings{bian2019neurips,
  title={Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video},
  author={Bian, Jiawang and Li, Zhichao and Wang, Naiyan and Zhan, Huangying and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

#### SC-DepthV2:
**Auto-Rectify Network for Unsupervised Indoor Depth Estimation (TPAMI 2022)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun Chin, Chunhua Shen, Ian Reid*
[**[paper]**](https://arxiv.org/abs/2006.02708v2)
```
@article{bian2021tpami, 
  title={Auto-Rectify Network for Unsupervised Indoor Depth Estimation}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Chin, Tat-Jin and Shen, Chunhua and Reid, Ian}, 
  journal= {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
  year={2021} 
}
```

#### SC-DepthV3:
**SC-DepthV3: Robust Self-supervised Monocular Depth Estimation for Dynamic Scenes (ArXiv 2022)** \
*Libo Sun\*, Jia-Wang Bian\*, Huangying Zhan, Wei Yin, Ian Reid, Chunhua Shen*
[**[paper]**](https://arxiv.org/abs/2211.03660) \
\* denotes equal contribution and joint first author
```
@article{sc_depthv3, 
  title={SC-DepthV3: Robust Self-supervised Monocular Depth Estimation for Dynamic Scenes}, 
  author={Sun, Libo and Bian, Jia-Wang and Zhan, Huangying and Yin, Wei and Reid, Ian and Shen, Chunhua}, 
  journal= {arXiv:2211.03660}, 
  year={2022} 
}
```
