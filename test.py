import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from datasets.test_folder import TestSet
from losses.loss_functions import compute_errors
from SC_Depth import SC_Depth
from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = SC_Depth(hparams)

    # load ckpts
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    # get training resolution
    training_size = get_training_size(hparams.dataset_name)

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
        pred_depth = model(tgt_img.cuda())

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
