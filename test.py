import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# pytorch-lightning
from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

from datasets.test_folder import TestSet
import datasets.custom_transforms as custom_transforms

from losses.loss_functions import compute_errors

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    model = system.load_from_checkpoint(hparams.ckpt_path)
    model.cuda()
    model.eval()

    # dataset
    if hparams.dataset_name == 'nyu':
        training_size = [256, 320]
    elif hparams.dataset_name == 'kitti':
        training_size = [256, 832]
    elif hparams.dataset_name == 'ddad':
        training_size = [384, 640]

    # data loader
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )
    test_dataset = TestSet(
        hparams.dataset_dir,
        transform=test_transform,
        dataset=hparams.dataset_name
    )
    print('{} samples found in test scenes'.format(len(test_dataset)))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             )

    all_errs = []
    for i, (tgt_img, gt_depth) in enumerate(tqdm(test_loader)):
        pred_depth = model.inference_depth(tgt_img.cuda())

        errs = compute_errors(gt_depth.cuda(), pred_depth,
                              hparams.dataset_name)

        all_errs.append(np.array(errs))

    all_errs = np.stack(all_errs)
    mean_errs = np.mean(all_errs, axis=0)

    print("\n  " + ("{:>8} | " * 9).format("abs_diff", "abs_rel",
          "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 9).format(*mean_errs.tolist()) + "\\\\")


if __name__ == '__main__':
    main()
