# This script is used to filter out static-camera frames. We determine the
# camera movement by detecting points in image boarders (often dynamic objects
# are not located there) and computing their LK optical flow. We regard the
# camera as static, if the average pixel shift is smaller than a threshold.

import numpy as np
import cv2
from path import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Selecting video frames for training sc_depth')
    parser.add_argument('--dataset_dir', required=True)
    args = parser.parse_args()
    return args


def compute_pixel_shift(frame1, frame2, feature_params, lk_params):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    h, w = frame1_gray.shape
    mask_features = np.zeros_like(frame1_gray)
    mask_features[:, 0:int(w*0.05)] = 1
    mask_features[:, int(w*0.95):w] = 1

    p0 = cv2.goodFeaturesToTrack(
        frame1_gray, mask=mask_features, **feature_params)
    p1, st, errs = cv2.calcOpticalFlowPyrLK(
        frame1_gray, frame2_gray, p0, None, **lk_params
    )

    if np.sum(st == 1) > 0:
        err = np.median(errs[st == 1])
    else:
        err = 1000000

    return err


def generate_index(scene):

    images = sorted(scene.files('*.jpg'))

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000, qualityLevel=0.01,
                          minDistance=8, blockSize=19)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    index = [0]
    errs = []
    for idx in range(1, len(images)):

        frame1 = cv2.imread(images[index[-1]])
        frame2 = cv2.imread(images[idx])

        h, w = frame1.shape[:2]
        l = np.linalg.norm(np.array([h, w]))
        err = compute_pixel_shift(frame1, frame2, feature_params, lk_params)

        if err < l * 0.01:
            continue

        errs.append(err)
        index.append(idx)

    print(len(images), len(index))
    return index


def main():

    args = parse_args()

    DataRoot = Path(args.dataset_dir)

    scenes = sorted((DataRoot/'training').dirs())
    for scene in scenes:
        print(scene)
        index = generate_index(scene)
        np.savetxt(scene/'frame_index.txt', index, fmt='%d', delimiter='\n')


if __name__ == '__main__':
    main()
