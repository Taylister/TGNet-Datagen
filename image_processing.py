import os
import shutil
import numpy as np
import cv2

from skimage import io
from skimage.morphology import skeletonize
from skimage.transform import resize

from argparse import ArgumentParser

class ImageProcess:
    """
    Reduce noise and fill closed resion of mask
    """
    def __init__(self,CSNet_img_path,title_img_path,output_path):
        self.CSNet_img_path = CSNet_img_path
        self.title_img_path = title_img_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)


        self.target_list = []

        name_list = os.listdir(self.CSNet_img_path)

        for filename in name_list:
            target_full_path = os.path.join(self.CSNet_img_path, filename)
            self.target_list.append(target_full_path)

    def _noise_reduction(self, kernel_size=3):
        
        savedir = os.path.join(self.output_path,"processed_mask")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        self.processed_target_list = []

        for path in self.target_list:
            img_name = os.path.basename(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            processed_img = cv2.medianBlur(img, kernel_size)

            ret, th = cv2.threshold(processed_img, 0, 255, cv2.THRESH_OTSU)

            save_path = os.path.join(savedir,img_name)
            self._save_image(save_path,th)

            self.processed_target_list.append(save_path)
    
    def _skeletonize(self):

        savedir = os.path.join(self.output_path,"skeletonize")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        for path in self.processed_target_list:
            img_name = os.path.basename(path)

            # read binarized image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            #binarize
            img[img == 255] = 1
            img[img != 1] = 0
        
            #skeletonize
            skeleton = skeletonize(img)

            output = np.where(skeleton, 255, 0)

            save_path = os.path.join(savedir,img_name)
            self._save_image(save_path,output)

    def _extract_mask_region(self):

        mat_savedir = os.path.join(self.output_path,"extracted_title")
        resized_savedir = os.path.join(self.output_path,"resized_title")
        
        if not os.path.exists(mat_savedir):
            os.mkdir(mat_savedir)
        
        if not os.path.exists(resized_savedir):
            os.mkdir(resized_savedir)
        
        for path in self.processed_target_list:
            img_name = os.path.basename(path)

            # read processed mask image
            msk = cv2.imread(path)

            # read original title image and resize
            title_img_path = os.path.join(self.title_img_path,img_name)
            
            title = io.imread(title_img_path)
            h, w = msk.shape[:2]
        
            to_h = h
            to_w = w
            to_scale = (to_h, to_w)
        
            resized_title = resize(title, to_scale, preserve_range=True)

            io.imsave(os.path.join(resized_savedir,img_name),resized_title)

            # read resized title as cv2 format

            resized_title = cv2.imread(os.path.join(resized_savedir,img_name))

            blend = cv2.bitwise_and(resized_title,msk)

            save_path = os.path.join(mat_savedir,img_name)
            self._save_image(save_path,blend)
            

    def _save_image(self, save_path, src):
        cv2.imwrite(save_path,src)


    def run(self):
        """
        Main function of this class
        """
        self._noise_reduction()

        self._skeletonize()

        self._extract_mask_region()


def main(args):
    Image_Process = ImageProcess(args.CSNet_img_path,args.title_img_path,args.output_path)
    Image_Process.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'CSNet_img_path',
        type=str,
        help='path to directory including output of CSNet'
    )
    parser.add_argument(
        'title_img_path',
        type=str,
        help='path to directory including title part image of book cover'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help=('path to directory processed images are saved in')
    )
    args = parser.parse_args()
    main(args)
