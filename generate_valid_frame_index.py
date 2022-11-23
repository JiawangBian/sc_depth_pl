# This script is used to filter out static-camera frames.
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


def compute_movement_ratio(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h, w = frame1_gray.shape
    diff = np.abs(frame1_gray - frame2_gray)
    ratio = (diff > 10).sum() / (h*w)
    return ratio


def generate_index(scene):

    images = sorted(scene.files('*.jpg'))

    index = [0]
    for idx in range(1, len(images)):

        frame1 = cv2.imread(images[index[-1]])
        frame2 = cv2.imread(images[idx])

        move_ratio = compute_movement_ratio(frame1, frame2)
        if move_ratio < 0.5:
            continue
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
