import os
import shutil
import numpy as np
import pandas as pd

import cv2
import pygame

from pygame import freetype
from skimage import io
from skimage.morphology import skeletonize
from skimage.transform import resize

from argparse import ArgumentParser

def main(args):

    original_df = pd.read_csv(args.original_csv_path)
    processed_df = pd.read_csv(args.processed_csv_path)
    
    book_IDs = processed_df["bookID"]

    for book_id in book_IDs:
        rec_text1 = original_df.query('bookID==@book_id').iloc[0,1]
        rec_text2 = processed_df.query('bookID==@book_id').iloc[0,1]

        if rec_text1 != rec_text2:
            #print("book_ID:{}".format(book_id))
            #print("original:{} processed:{}".format(rec_text1,rec_text2))
            delete_images(args.tg_data_dirpath, book_id)

def delete_images(dirpath,image_name):

    if os.path.exists(dirpath):
        paths = os.listdir(dirpath)

        for path in paths:
            target_dirpath = os.path.join(dirpath,path)
            target_filepath = os.path.join(target_dirpath,image_name)

            if os.path.isfile(target_filepath):
                os.remove(target_filepath)
                continue
            else:
                continue
    else:
        raise NameError("such file is not exist.")
  

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'tg_data_dirpath',
        type=str,
        help='path to directory including tg-data'
    )
    parser.add_argument(
        'original_csv_path',
        type=str,
        help='path to csv including the text information of extracted title region (from original)'
    )
    parser.add_argument(
        'processed_csv_path',
        type=str,
        help='path to csv including the text information of extracted title region (from processed)'
    )
    args = parser.parse_args()
    main(args)
