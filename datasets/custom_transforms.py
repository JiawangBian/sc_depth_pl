from __future__ import division
import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics=None):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            shape = tensor.size()
            if shape[0] == 3:
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            if im.ndim < 3:  # depth
                im = np.expand_dims(im, axis=0)
                tensors.append(torch.from_numpy(im).float())
            else:
                im = np.transpose(im, (2, 0, 1))
                tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        scaled_images = []
        for im in images:
            if im.ndim < 3:  # depth
                scaled_images.append(cv2.resize(im, dsize=(
                    scaled_w, scaled_h), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST))
            else:
                scaled_images.append(cv2.resize(im, dsize=(
                    scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR))

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h,
                             offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics


class RescaleTo(object):
    """Rescale images to training or validation """

    def __init__(self, output_size=[256, 832]):
        self.output_size = output_size

    def __call__(self, images, intrinsics):

        in_h, in_w, _ = images[0].shape
        out_h, out_w = self.output_size[0], self.output_size[1]

        if in_h != out_h or in_w != out_w:

            scaled_images = []
            for im in images:
                if im.ndim < 3:  # depth
                    scaled_images.append(cv2.resize(im, dsize=(
                        out_w, out_h), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST))
                else:
                    scaled_images.append(cv2.resize(im, dsize=(
                        out_w, out_h), interpolation=cv2.INTER_LINEAR))
        else:
            scaled_images = images

        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)
            output_intrinsics[0] *= (out_w * 1.0 / in_w)
            output_intrinsics[1] *= (out_h * 1.0 / in_h)
        else:
            output_intrinsics = None

        return scaled_images, output_intrinsics
