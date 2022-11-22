import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from path import Path
from imageio.v2 import imread
from scipy import sparse

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluation scripts")
parser.add_argument("--dataset", required=True, help="kitti or nyu",
                    choices=['nyu', 'bonn', 'tum', 'kitti', 'ddad', 'scannet'], type=str)
parser.add_argument("--pred_depth", required=True,
                    help="predicted depth folders", type=str)
parser.add_argument("--gt_depth", required=True,
                    help="gt depth folders", type=str)
parser.add_argument("--seg_mask", default=None,
                    help="segmentation mask folders", type=str)

######################################################
args = parser.parse_args()


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense())
    return depth


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3


class DepthEval():
    def __init__(self):

        self.min_depth = 0.1

        if args.dataset == 'nyu':
            self.max_depth = 10.
        elif args.dataset == 'scannet':
            self.max_depth = 10.
        elif args.dataset == 'bonn':
            self.max_depth = 10.
        elif args.dataset == 'tum':
            self.max_depth = 10.
        elif args.dataset == 'kitti':
            self.max_depth = 80.
        elif args.dataset == 'ddad':
            self.max_depth = 200.

    def main(self):
        pred_depths = []

        """ Get results """
        pred_depths = sorted(Path(args.pred_depth).files("*.npy"))  # in *.npy

        """ get gt depths """
        if args.dataset in ['nyu', 'scannet', 'bonn', 'tum']:
            gt_depths = sorted(Path(args.gt_depth).files("*.png"))  # in *.png
        elif args.dataset == 'kitti':
            gt_depths = sorted(Path(args.gt_depth).files("*.npy"))  # in *.npy
        elif args.dataset == 'ddad':
            gt_depths = sorted(Path(args.gt_depth).files("*.npz"))  # in *.npz
        else:
            print('the datset is not support')

        assert (len(pred_depths) == len(gt_depths))

        """ Get segmentation masks """
        seg_masks = None
        if args.seg_mask is not None:
            self.dynamic_colors = np.loadtxt(
                Path(args.seg_mask)/'dynamic_colors.txt').astype('uint8')
            seg_masks = sorted(Path(args.seg_mask).files("*.png"))

        self.evaluate_depth(gt_depths, pred_depths, seg_masks, eval_mono=True)

    def evaluate_depth(self, gt_depths, pred_depths, seg_masks=None, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths: list of gt depth files
            pred_depths: list of predicted depth files
            eval_mono (bool): use median scaling if True
        """
        full_errors = []
        static_errors = []
        dynamic_errors = []
        ratios = []

        print("==> Evaluating depth result...")
        for i in tqdm(range(len(pred_depths))):
            # load predicted depth
            pred_depths[i] = np.load(pred_depths[i])

            # load gt depth
            if args.dataset in ['nyu']:
                gt_depths[i] = imread(gt_depths[i]).astype(np.float32) / 5000
            elif args.dataset in ['scannet', 'bonn', 'tum']:
                gt_depths[i] = imread(gt_depths[i]).astype(np.float32) / 1000
            elif args.dataset == 'kitti':
                gt_depths[i] = np.load(gt_depths[i])
            elif args.dataset == 'ddad':
                gt_depths[i] = load_sparse_depth(gt_depths[i])
            else:
                print('the datset is not support')

            # load seg mask
            if seg_masks is not None:
                dynamic_mask = np.zeros_like(gt_depths[i])
                seg_mask = imread(seg_masks[i])
                for item in self.dynamic_colors:
                    cal_mask_0 = seg_mask[:, :, 0] == item[0]
                    cal_mask_1 = seg_mask[:, :, 1] == item[1]
                    cal_mask_2 = seg_mask[:, :, 2] == item[2]
                    cal_mask = cal_mask_0 * cal_mask_1 * cal_mask_2
                    dynamic_mask[cal_mask] = 1

            # gt
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            mask = np.logical_and(gt_depth > self.min_depth,
                                  gt_depth < self.max_depth)

            # # resize predicted depth to gt resolution
            pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

            # pre-process
            if args.dataset == 'kitti':
                crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            elif args.dataset == 'nyu':
                crop = np.array([45, 471, 41, 601]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            val_pred_depth = pred_depth[mask]
            val_gt_depth = gt_depth[mask]

            # median scaling is used for monocular evaluation
            ratio = 1
            if eval_mono:
                ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                ratios.append(ratio)
                val_pred_depth *= ratio

            val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
            val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

            full_errors.append(compute_depth_errors(
                val_gt_depth, val_pred_depth))

            if seg_masks is not None:
                val_dynamic_mask = dynamic_mask[mask]

                # every image has static regions
                static_errors.append(compute_depth_errors(val_gt_depth[val_dynamic_mask == 0],
                                                          val_pred_depth[val_dynamic_mask == 0]))

                # note that some images may not have dynamic regions,
                # we only average results on images that have dynamic regions
                if (val_gt_depth[val_dynamic_mask == 1]).shape[0] > 0:
                    full_errors.append(compute_depth_errors(
                        val_gt_depth, val_pred_depth))
                    dynamic_errors.append(compute_depth_errors(val_gt_depth[val_dynamic_mask == 1],
                                                               val_pred_depth[val_dynamic_mask == 1]))

            pred_depths[i] = None

        if eval_mono:
            ratios = np.array(ratios)
            print(
                " Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))

        mean_errors_full = np.array(full_errors).mean(0)
        print("Evaluation on full images")
        print("\n " + ("{:>8} | " * 8).format("abs_rel", "sq_rel",
              "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 8).format(*mean_errors_full.tolist()) + "\\\\")

        if seg_masks is not None:
            print("\n Evaluation on dynamic regions")
            mean_errors_full = np.array(dynamic_errors).mean(0)
            print("\n  " + ("{:>8} | " * 8).format("abs_rel",
                  "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 8).format(*
                  mean_errors_full.tolist()) + "\\\\")

            print("\n Evaluation on static regions")
            mean_errors_full = np.array(static_errors).mean(0)
            print("\n  " + ("{:>8} | " * 8).format("abs_rel",
                  "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 8).format(*
                  mean_errors_full.tolist()) + "\\\\")


eval = DepthEval()
eval.main()
