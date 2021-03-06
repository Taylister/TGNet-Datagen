import os
import shutil
import numpy as np
import pandas as pd

import cv2

from skimage import io
from skimage.morphology import skeletonize
from skimage.transform import resize

from tqdm import tqdm

from argparse import ArgumentParser

def main(args):
    exclude = ['raw_mask','title','original.csv','processed.csv']

    if not os.path.exists(args.deployed_dirpath):
        os.mkdir(args.deployed_dirpath)

    dirs = os.listdir(args.data_dirpath)
    pbar = tqdm(len(dirs),total=len(dirs))
    pbar.set_description("Copying dataset..")
    for dir in dirs:
        pbar.update(1)
        if not (dir in exclude):
            original_path = os.path.join(args.data_dirpath,dir)
            duplicate_path = os.path.join(args.deployed_dirpath,dir)
            shutil.copytree(original_path,duplicate_path)
        else:
            continue
    pbar.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'data_dirpath',
        type=str,
        help='path to directory including the real data for training TGNet'
    )
    parser.add_argument(
        'deployed_dirpath',
        type=str,
        help='path to directory the data for training TGNet deployed'
    )
    args = parser.parse_args()
    main(args)