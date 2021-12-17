import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

import datasets.custom_transforms as custom_transforms
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet
from losses.loss_functions import (compute_errors, compute_smooth_loss,
                                   photo_and_geometry_loss)
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from visualization import *


class SC_Depth(LightningModule):
    def __init__(self, hparams):
        super(SC_Depth, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()

    def prepare_data(self):
        if self.hparams.hparams.dataset_name == 'kitti':
            training_size = [256, 832]
        elif self.hparams.hparams.dataset_name == 'ddad':
            training_size = [384, 640]
        else:
            print('unknown dataset type')

        # data loader
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.RescaleTo(training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        valid_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

        self.train_dataset = TrainFolder(
            self.hparams.hparams.dataset_dir,
            transform=train_transform,
            train=True,
            sequence_length=self.hparams.hparams.sequence_length,
            skip_frames=self.hparams.hparams.skip_frames,
            dataset=self.hparams.hparams.dataset_name
        )

        if self.hparams.hparams.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=valid_transform,
                dataset=self.hparams.hparams.dataset_name
            )
        elif self.hparams.hparams.val_mode == 'photo':
            self.val_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                transform=valid_transform,
                train=False,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name
            )
        else:
            print("wrong validation mode")

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validatioin'.format(len(self.val_dataset)))

    def inference_depth(self, img):
        pred_depth = self.depth_net(img)
        return pred_depth

    def inference_pose(self, img1, img2):
        pred_pose = self.pose_net(img1, img2)
        return pred_pose

    def forward(self, tgt_img, ref_imgs):
        # in lightning, forward defines the prediction/inference actions for training
        tgt_depth = self.inference_depth(tgt_img)
        ref_depths = [self.inference_depth(im) for im in ref_imgs]

        poses = [self.inference_pose(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.inference_pose(im, tgt_img) for im in ref_imgs]

        return tgt_depth, ref_depths, poses, poses_inv

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        tgt_img, ref_imgs, intrinsics = batch

        # network forward
        tgt_depth, ref_depths, poses, poses_inv = self(tgt_img, ref_imgs)

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight

        loss_1, loss_2 = photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                 intrinsics, poses, poses_inv, self.hparams.hparams)
        loss_3 = compute_smooth_loss(tgt_depth, tgt_img)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = self.inference_depth(tgt_img)
            errs = compute_errors(gt_depth, tgt_depth,
                                  self.hparams.hparams.dataset_name)

            errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                    'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs, intrinsics = batch
            tgt_depth, ref_depths, poses, poses_inv = self(tgt_img, ref_imgs)
            loss_1, loss_2 = photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                     intrinsics, poses, poses_inv)
            errs = {'photo_loss': loss_1.item(), 'geometry_loss': loss_2.item()}
        else:
            print('wrong validation mode')

        if self.global_step < 10:
            return errs

        # plot
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0])  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[0, 0])  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(
                0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)

        return errs

    def validation_epoch_end(self, outputs):

        if self.hparams.hparams.val_mode == 'depth':
            mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in outputs]).mean()

            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)

        elif self.hparams.hparams.val_mode == 'photo':
            mean_pl = np.array([x['photo_loss'] for x in outputs]).mean()
            self.log('val_loss', mean_pl, prog_bar=True)
