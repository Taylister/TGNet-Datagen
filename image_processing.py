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
from tqdm import tqdm

import render_standard_text

from argparse import ArgumentParser

class ImageProcess:
    """
    Reduce noise and fill closed resion of mask
    """
    def __init__(   self,
                    CSNet_img_dirpath,
                    title_img_dirpath,
                    standard_font_path,
                    original_csv_path,
                    output_path):

        self.CSNet_img_dirpath = CSNet_img_dirpath
        self.title_img_dirpath = title_img_dirpath
        self.standard_font_path = standard_font_path
        self.original_csv_path = original_csv_path

        self.output_path = output_path

        freetype.init()

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)


        self.target_list = []

        name_list = os.listdir(self.CSNet_img_dirpath)

        for filename in name_list:
            target_full_path = os.path.join(self.CSNet_img_dirpath, filename)
            self.target_list.append(target_full_path)

    def _noise_reduction(self, kernel_size=3):
        
        savedir = os.path.join(self.output_path,"processed_mask")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        self.processed_target_list = []

        pbar = tqdm(len(self.target_list),total=len(self.target_list))
        pbar.set_description("Noise reduction")
        for path in self.target_list:
            img_name = os.path.basename(path)
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            sq_img = self._to_square(img)
            
            sq_img = cv2.cvtColor(sq_img,cv2.COLOR_BGR2GRAY)
            processed_img = cv2.medianBlur(sq_img, kernel_size)
            ret, th = cv2.threshold(processed_img, 0, 255, cv2.THRESH_OTSU)

            save_path = os.path.join(savedir,img_name)
            self._save_image(save_path,th)

            self.processed_target_list.append(save_path)

            pbar.update(1)
        
        pbar.close()
    
    def _skeletonize(self):

        savedir = os.path.join(self.output_path,"skeletonize")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        pbar = tqdm(len(self.processed_target_list),total=len(self.processed_target_list))
        pbar.set_description("Skletonize")
        for path in self.processed_target_list:
            img_name = os.path.basename(path)

            # read binarized image
            processed_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ret, th = cv2.threshold(processed_mask, 0, 255, cv2.THRESH_OTSU)

            #binarize
            th[th > 0] = 1
        
            #skeletonize
            skeleton = skeletonize(th)

            output = np.where(skeleton, 255, 0)

            save_path = os.path.join(savedir,img_name)
            self._save_image(save_path,output)
            pbar.update(1)
        pbar.close()

    def _extract_mask_region(self):

        mat_savedir = os.path.join(self.output_path,"extracted_title")
        resized_savedir = os.path.join(self.output_path,"resized_title")
        
        if not os.path.exists(mat_savedir):
            os.mkdir(mat_savedir)
        
        if not os.path.exists(resized_savedir):
            os.mkdir(resized_savedir)
        
        pbar = tqdm(len(self.processed_target_list),total=len(self.processed_target_list))
        pbar.set_description("Extract")
        for path in self.processed_target_list:

            img_name = os.path.basename(path)

            # read processed mask image
            msk = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            ret, msk = cv2.threshold(msk, 0, 255, cv2.THRESH_OTSU)
           
            # read original title image and save resized title image
            title_img_path = os.path.join(self.title_img_dirpath,img_name)
            resized_title = self._resize_image(64,title_img_path)

            io.imsave(os.path.join(resized_savedir,img_name), resized_title.astype(np.uint8),  check_contrast=False)

            # extract title region as backgroud color [128,128,128]

            blend = cv2.imread(os.path.join(resized_savedir,img_name),cv2.IMREAD_COLOR)
            blend = self._to_square(blend)        
            blend[msk==0] = [127,127,127]

            save_path = os.path.join(mat_savedir,img_name)
            self._save_image(save_path,blend)

            pbar.update(1)
        pbar.close()

    def _gen_text_source(self):

        book_df = pd.read_csv(self.original_csv_path)

        savedir = os.path.join(self.output_path,"input_text")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        pbar = tqdm(len(self.processed_target_list),total=len(self.processed_target_list))
        pbar.set_description("Generate i_t")
        for path in self.processed_target_list:
            img_name = os.path.basename(path)
            im = cv2.imread(path)
            # surf_h, surf_w = im.shape[0],im.shape[1]
            surf_h, surf_w = 256,256
            
            text = book_df.query('bookID==@img_name').iloc[0,1]
            text = str(text)

            i_t = render_standard_text.make_standard_text(self.standard_font_path,text,(surf_h,surf_w))

            save_path = os.path.join(savedir,img_name)
            self._save_image(save_path,i_t)
            pbar.update(1)
        pbar.close()

    def _resize_image(self, rescale_h, img_path):      
        title = io.imread(img_path)
        h, w = title.shape[:2]
        scale_ratio = rescale_h / h
        to_h = rescale_h
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
            
        to_scale = (to_h, to_w)

        return resize(title, to_scale, preserve_range=True)
            
    def _to_square(self, img):
        height, width = img.shape[:2]
        tmp = img[:, :]
        if(height > width):
            size = height
            limit = width
        else:
            size = width
            limit = height

        start = int((size - limit) / 2)
        fin = int((size + limit) / 2)
        sq_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
            
        if(size == height):
            sq_img[:,start:fin]=tmp
            return sq_img
        else:
            sq_img[start:fin,:]=tmp
            return sq_img

    def _save_image(self, save_path, src):
        cv2.imwrite(save_path,src)


    def run(self):
        """
        Main function of this class
        """
        if not os.path.exists(os.path.join(self.output_path,"processed_mask")):
            self._noise_reduction()
        else:
            print("#Reducing noise of the mask image is already done")
            
            self.processed_target_list = []

            filename_list = os.listdir(os.path.join(self.output_path,"processed_mask"))
            for filename in filename_list:
                filepath = os.path.join(self.output_path,"processed_mask",filename)
                self.processed_target_list.append(filepath)

        if not os.path.exists(os.path.join(self.output_path,"skeletonize")):
            self._skeletonize()
        else:
            print("#Skeletonization of processed mask image is already done")

        if not os.path.exists(os.path.join(self.output_path,"extracted_title")):
            self._extract_mask_region()
        else:
            print("#Extraction of character region of the title image is already done")
        
        #self._select_proper_image()
        if not os.path.exists(os.path.join(self.output_path,"input_text")):
            self._gen_text_source()
        else:
            print("#Generating input text image of TGnet is already done")



def main(args):
    Image_Process = ImageProcess(   args.CSNet_img_dirpath,
                                    args.title_img_dirpath,
                                    args.standard_font_path,
                                    args.original_csv_path,
                                    args.output_path)
    Image_Process.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'CSNet_img_dirpath',
        type=str,
        help='path to directory including output of CSNet'
    )
    parser.add_argument(
        'title_img_dirpath',
        type=str,
        help='path to directory including title part image of book cover'
    )
    parser.add_argument(
        'standard_font_path',
        type=str,
        help='path to .ttf file rendered on input text image '
    )
    parser.add_argument(
        'original_csv_path',
        type=str,
        help='path to csv including the text information of extracted title region'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help=('path to directory processed images are saved in')
    )
    args = parser.parse_args()
    main(args)
