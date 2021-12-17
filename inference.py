import numpy as np
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os

from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

import datasets.custom_transforms as custom_transforms

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)

    model = system.load_from_checkpoint(hparams.ckpt_path)
    model.cuda()
    model.eval()

    # training size
    if hparams.dataset_name == 'nyu':
        training_size = [256, 320]
    elif hparams.dataset_name == 'kitti':
        training_size = [256, 832]
    elif hparams.dataset_name == 'ddad':
        training_size = [384, 640]

    # normaliazation
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir) / \
        'model_{}'.format(hparams.model_version)
    output_dir.makedirs_p()

    if hparams.save_vis:
        (output_dir/'vis').makedirs_p()

    if hparams.save_depth:
        (output_dir/'depth').makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    for i, img_file in enumerate(tqdm(image_files)):

        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        pred_depth = model.inference_depth(tensor_img)

        if hparams.save_vis:
            vis = visualize_depth(pred_depth[0, 0]).permute(
                1, 2, 0).numpy() * 255
            imwrite(output_dir/'vis/{}.jpg'.format(filename),
                    vis.astype(np.uint8))

        if hparams.save_depth:
            depth = pred_depth[0, 0].cpu().numpy()
            np.save(output_dir/'depth/{}.npy'.format(filename), depth)


if __name__ == '__main__':
    main()
