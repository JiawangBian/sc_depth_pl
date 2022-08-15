from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import datasets.custom_transforms as custom_transforms
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet


class VideosDataModule(LightningDataModule):

    def __init__(self, dataset_dir,
                 dataset_name='kitti',
                 training_size=[256, 320],
                 sequence_length=3,
                 skip_frames=1,
                 batch_size=4,
                 val_mode='depth'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.skip_frames = skip_frames
        self.training_size = training_size
        self.batch_size = batch_size
        self.val_mode = val_mode
        self.dataset_name = dataset_name

        # data loader
        self.train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        self.valid_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.train_dataset = TrainFolder(
            self.dataset_dir,
            transform=self.train_transform,
            train=True,
            sequence_length=self.sequence_length,
            skip_frames=self.skip_frames
        )

        if self.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.dataset_dir,
                transform=self.valid_transform,
                dataset=self.dataset_name
            )
        elif self.val_mode == 'photo':
            self.val_dataset = TrainFolder(
                self.dataset_dir,
                transform=self.valid_transform,
                train=False,
                sequence_length=self.sequence_length,
                skip_frames=self.skip_frames
            )
        else:
            print("wrong validation mode")

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validatioin'.format(len(self.val_dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.batch_size,
                          pin_memory=True)
