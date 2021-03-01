#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import cv2


def storeBinaryImage(gray_img):
    """
    Return a binary image as a 2-dimensional numpy array.
    """
    return cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


def getRectFrom4Points(pts):
    """
    Return a rectangle information about the text region.
    """
    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    return [x_min, y_min, x_max-x_min, y_max-y_min]

def searchLargestTextRegion(pts_matrix):
    """
    Return the index, which have most largest text region
    """
    rects = []

    for pts in pts_matrix:
        rect = getRectFrom4Points(pts)
        rects.append(rect)

    max_ssize = 0
    key = 0

    for index, rect in enumerate(rects):
        rect_width = rect[2]
        rect_height = rect[3]
        if max_ssize <= (rect_width * rect_height):
            max_ssize = rect_height * rect_width
            key = index

    return key


def getRectImage(bin_img, rect):
    """
    Return a image which is cut based on the rectangle of the character.
    """
    height, width = bin_img.shape[0],bin_img.shape[1]
    x, y, rect_width, rect_height = rect
    x_min = x - 1 if x > 0 else x 
    y_min = y - 1 if y > 0 else y
    x_max = x_min + rect_width + 1 if x_min + rect_width <= width else x_min + rect_width
    y_max = y_min + rect_height + 1 if y_min + rect_height <= height else y_min + rect_height
    segmented_img = bin_img[y_min : y_max, x_min : x_max]
    return segmented_img


def getSquareImage(bin_img):
    """
    Return a square image being padded by white pixels.
    """
    height, width = bin_img.shape
    if height == width:
        return bin_img
    elif height > width:
        square_img = np.empty((height, height), dtype=np.uint8)
        square_img[:, :] = 255
        half_length = int((height - width) / 2)
        if half_length % 2 == 0:
            square_img[0 : height, half_length : half_length+width] = bin_img[0 : height, 0 : width]
        else:
            square_img[0 : height, half_length+1 : half_length+1+width] = bin_img[0 : height, 0 : width]
        return square_img
    else:
        square_img = np.empty((width, width), dtype=np.uint8)
        square_img[:, :] = 255
        half_length = int((width - height) / 2)
        if half_length % 2 == 0:
            square_img[half_length : half_length+height, 0 : width] = bin_img[0 : height, 0 : width]
        else:
            square_img[half_length+1 : half_length+1+height, 0 : width] = bin_img[0 : height, 0 : width]
        return square_img


def isWhiteBackground(bin_img):
    """
    If the backgroud color of input binary image seems white, return True.
    """
    height, width = bin_img.shape
    white_cols = np.where(np.sum(bin_img, axis=0) == height * 255)[0]
    if len(white_cols) > 0:
        return True


def countBlackCorner(bin_img):
    """
    Return the number of corners whose pixel value is 0.
    I.e., 0 <= (the output number) <= 4.
    """
    height, width = bin_img.shape
    counter = 0
    if bin_img[0][0] == 0:
        counter += 1
    if bin_img[0][width - 1] == 0:
        counter += 1
    if bin_img[height - 1][0] == 0:
        counter += 1
    if bin_img[height - 1][width - 1] == 0:
        counter += 1
    return counter


def isBlackBackground(bin_img, judge_thresh=0.7):
    """
    If the backgroud color of input binary image seems black, return True.
    """
    height, width = bin_img.shape
    
    white_pixel_num = np.count_nonzero(bin_img)
    black_pixel_num = height * width - white_pixel_num
    if black_pixel_num > height * width * judge_thresh:
        return True
    
    black_cols = np.where(np.sum(bin_img, axis=0) == 0)[0]
    black_rows = np.where(np.sum(bin_img, axis=1) == 0)[0]
    if len(black_rows) > 0 or len(black_cols) > 0:
        return True
    
    if countBlackCorner(bin_img) > 2:
        return True
    
    return False


def isTooWhite(bin_img, judge_thresh=0.98):
    """
    If the input binary image has too much white pixels, return True.
    """
    height, width = bin_img.shape
    white_pixel_num = np.count_nonzero(bin_img)
    if white_pixel_num > height * width * judge_thresh:
        return True
    else:
        return False


def isNoisy(bin_img, value_change_thresh=30):
    """
    If the input binary image has too much edges, return True.
    """
    height, width = bin_img.shape
    vertical_value_change_num = horizontal_value_change_num = 0
    for x in range(width):
        value_change_counter = 0
        first_pixel_value = bin_img[0][x]
        for y in range(height):
            if bin_img[y][x] != first_pixel_value:
                value_change_counter += 1
                first_pixel_value = bin_img[y][x]
        vertical_value_change_num += value_change_counter
    vertical_value_change_mean = int(vertical_value_change_num / height)
    if vertical_value_change_mean > value_change_thresh:
        return True
    
    for y in range(height):
        value_change_counter = 0
        first_pixel_value = bin_img[y][0]
        for x in range(width):
            if bin_img[y][x] != first_pixel_value:
                value_change_counter += 1
                first_pixel_value = bin_img[y][x]
        horizontal_value_change_num += value_change_counter
    horizontal_value_change_mean = int(horizontal_value_change_num / width)
    if horizontal_value_change_mean > value_change_thresh:
        return True
    
    return False


class Segmentation:
    """
    Detect and segment text regions, then split those into each character.
    And generate mask and inpaint image for network training.
    """
    height_thresh = 20

    def __init__(self, img_filepath, txt_filepath, resized_length=100):
        self.img_filepath = img_filepath
        self.txt_filepath = txt_filepath
        self.resized_length = resized_length
        self.non_text_region_pts = []
        self.text_region_pts = []
        self.text_region_imgs = []
        self.char_imgs = []
        self.char_img_indices = []
        self.masked_imgs = []
        self.inpainted_imgs = []
        self.all_mask = []



    def getTextRegionPointsFromText(self):
        """
        Return a np.array whose shape is (#text lines, 4, 2).
        """
        with open(self.txt_filepath) as f:
            lines = f.readlines()

        # pts_matrix = np.empty((len(lines), 8), int)
        pts_matrix = []
        for i, line in enumerate(lines):
            
            line = line.split('\n')[0]
            if '-' in line:
                continue
            str_nums = line.split(',')
            pts = []
            for j, str_num in enumerate(str_nums):
                pts.extend([int(str_num)])
                # pts_matrix[i][j] = int(str_num)
            pts_matrix.append(pts)
        
        pts_matrix = np.array(pts_matrix, np.int32)
       
        return pts_matrix.reshape((-1, 4, 2))


    def segmentTextRegions(self):
        """
        Segment text regions based on CRAFT.
        """

        self.img = cv2.imread(self.img_filepath)

        try:
            # TODO:imreadで読み込めない際の対応をスマートに書く
            self.height, self.width = self.img.shape[:2]
        except Exception as e:
            self.deleteFlag = True
            return

        # Store 4 coordinates of the text region bounding boxes
        pts_matrix = self.getTextRegionPointsFromText()
        # To make all inpainted image
        pts_matrix2 = pts_matrix
        
        # Noise reduction using filter that is able to preserve the edges of character
        filtered_img = cv2.edgePreservingFilter(self.img)

        gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    
        if (len(pts_matrix) > 0):
            while(len(self.masked_imgs) < 5):
                # when the detected points less than 5, break the loop.
                if(len(pts_matrix) <= len(self.masked_imgs)):
                    break

                # text region are picked desecendingly from largest text region size
                if(len(pts_matrix) > 1):
                    key = searchLargestTextRegion(pts_matrix)
                    
                else:
                    key = 0
                    
                rect = getRectFrom4Points(pts_matrix[key])
                text_region_img = getRectImage(filtered_img, rect)
                
                height, width = text_region_img.shape[0],text_region_img.shape[1]

                if height > 2 * width:
                    if len(pts_matrix) == 1:
                        # delete the image bacause probably the image sould be unsuitable for training
                        if len(text_region_img) == 0:
                            self.deleteFlag = True
                        
                        break

                    else:
                        # delete the elemetn bacause probably the points sould be unsuitable for training
                        # so that search next pts
                        pts_matrix = np.delete(pts_matrix,obj=key,axis=0)
                        continue

                if height * width <= 3000:
                    if len(pts_matrix) == 1:
                        # delete the image bacause the image sould be unsuitable for training
                        if len(text_region_img) == 0:
                            self.deleteFlag = True
                        
                        break

                    else:
                        # delete the elemetn bacause probably the points sould be unsuitable for training
                        # so that search next pts
                        pts_matrix = np.delete(pts_matrix,obj=key,axis=0)
                        continue
                
                # Generate mask image for inpainting
                mask_img = self.genMaskImage(pts_matrix[key])
                self.text_region_imgs.append(text_region_img)
                self.masked_imgs.append(mask_img)
                self.text_region_pts.append(pts_matrix[key])
                
                pts_matrix = np.delete(pts_matrix,obj=key,axis=0)

        else:
            # the process when text region isn't extracted.
            # generate no masked images(all black image)
            self.deleteFlag = True
        
        if len(self.text_region_imgs) > 0:
            for coordinates in pts_matrix2:
                mask = self.genMaskImage(coordinates)
                self.all_mask.append(mask)
       
            self.inpaintingAllMaskeRegions()

    def inpaintingAtMaskeRegions(self):
        """
        Inpainting input images by extracted text regions
        """
        for masked_img in self.masked_imgs:
            dst = cv2.inpaint(self.img,masked_img,3,cv2.INPAINT_NS)
            self.inpainted_imgs.append(dst)

    def inpaintingAllMaskeRegions(self):
        """
        Inpainting input images by extracted text regions
        """
        dst = self.img
        for mask in self.all_mask:
            dst = cv2.inpaint(dst,mask,3,cv2.INPAINT_NS)
            
        self.all_pts_inpainted_img = dst
            
    def drawTextRegions(self):
        """
        Draw rectangles of text regions.
        """
        for pts in self.text_region_pts:
            pts = pts.reshape(4, 1, 2)
            cv2.polylines(self.img, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        for pts in self.non_text_region_pts:
            pts = pts.reshape(4, 1, 2)
            cv2.polylines(self.img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    def storeResizedBoundingRectImage(self, white_bg_bin_img):
        """ Store a resized bounding rect image.
        * Find bounding rect of the input image
        * Do padding for making square image
        * Resize

        Args
        -------
        white_bg_bin_img: (ndarray of uint8)
            2D array of a binary image. The shape is (height, width).
        after_resize_length: (uint), optional
            Length of resized image.

        Returns
        -------
        resized_img: (ndarray of uint8)
            2D array of a resized image.
        rect: (list of uint)
            Bounding rectangle info (x, y, width, and height).
        """
        black_bg_bin_img = 255 - white_bg_bin_img
        x, y, w, h = cv2.boundingRect(black_bg_bin_img)
        if h < Segmentation.height_thresh:
            return None, None
        bounding_rect_img = white_bg_bin_img[y : y+h, x : x+w]

        total_padding_length = w - h
        one_side_padding_length = abs(total_padding_length) // 2
        if total_padding_length > 0:
            padded_img = np.empty((w, w), dtype=np.uint8)
            padded_img[:, :] = 255
            padded_img[one_side_padding_length : one_side_padding_length+h, :] = bounding_rect_img
        elif total_padding_length < 0:
            padded_img = np.empty((h, h), dtype=np.uint8)
            padded_img[:, :] = 255
            padded_img[:, one_side_padding_length : one_side_padding_length+w] = bounding_rect_img
        else:
            padded_img = bounding_rect_img
        resized_img = cv2.resize(padded_img, (self.resized_length, self.resized_length))
        return resized_img, [x, y, w, h]

    def segmentChars(self, labeling=True, cutting_thresh=0.98):
        """
        Segment each character from text region images.
        """
        if labeling:
            for i, img in enumerate(self.text_region_imgs):
                height, width = img.shape
                labeled_img = cv2.connectedComponents(255 - img)[1]

                # # Noise reduction by erotion and dilation
                # img_copy = img.copy()
                # img_copy = 255 - img_copy
                # kernel = np.ones( (3, 3), dtype=np.uint8 )
                # eroded_img = cv2.erode(img_copy, kernel, iterations=1)
                # dilated_img = cv2.dilate(eroded_img, kernel, iterations=2)
                # masked_img = cv2.bitwise_and(img_copy, img_copy, mask=dilated_img)
                # # Labeling
                # labeled_img = cv2.connectedComponents(masked_img)[1]

                if labeled_img.max() < 4:
                    continue
                x_img_dict = {}
                for label in range(1, labeled_img.max() + 1):
                    char_img = np.empty(img.shape, dtype=np.uint8)
                    char_img[:, :] = 255
                    char_img[labeled_img == label] = 0
                    resized_img, rect = self.storeResizedBoundingRectImage(char_img)
                    if resized_img is None:
                        continue
                    for j in range(rect[2]):
                        if not rect[0] + j in x_img_dict:
                            x_img_dict[rect[0] + j] = resized_img
                            break
                x_img_dict = sorted(x_img_dict.items())

                for x, char_img in x_img_dict:
                    self.char_imgs.append(char_img)
                    self.char_img_indices.append(i)
                    
        else: # Simple cutting algorithm
            for i, img in enumerate(self.text_region_imgs):
                height, width = img.shape
                
                # If (almost) all pixel of a column is white, x of that column becomes cutting 
                # candidates
                white_cols = np.where(np.sum(img, axis=0) >= height * 255 * cutting_thresh)[0]
                if len(white_cols) < 2:
                    continue
                
                # In below example, we only need 8 coordinates(x) of '|'(cutting line)
                # [        |h| |o| |g|        |e|     ]
                cutting_x = []
                index = 0
                if white_cols[index] != 0: # In case like [h| |o| |g|        |e|     ]
                    cutting_x.append(0)
                cutting_x.append(white_cols[index])
                index += 1
                if index != len(white_cols):
                    while index < len(white_cols) - 1:
                        if (white_cols[index] - 1 == white_cols[index - 1]
                            and white_cols[index] + 1 == white_cols[index + 1]):
                            index += 1
                        else:
                            cutting_x.append(white_cols[index])
                            index += 1
                cutting_x.append(white_cols[index])
                if white_cols[-1] != width - 1: # In case like [        |h| |o| |g|        |e]
                    cutting_x.append(width - 1)
                
                for j in range(len(cutting_x) - 1):
                    segmented_img = img[0 : height, cutting_x[j] : cutting_x[j+1]]
                    if isTooWhite(segmented_img):
                        continue
                    square_img = getSquareImage(segmented_img)
                    resized_img = cv2.resize(square_img, (self.resized_width, self.resized_height))
                    self.char_imgs.append(resized_img)
                    self.char_img_indices.append(i)
        self.char_imgs = np.array(self.char_imgs, dtype=np.uint8)
    
    def genMaskImage(self,pts):
        """
        Generate Mask Image to Inpaint
        """
        
        black_img = np.zeros((*self.img.shape[:-1],1), dtype=np.uint8)

        if pts == []:    
            # pts is empty,  this means there is no text region in the image
            # So, return not mask image(only black)
            return black_img
        else:
            return cv2.fillPoly(black_img,[pts],(255))
        
    def saveImage(self,img, dirpath,filename):
        """
        Save a image.
        """
        output_filepath = os.path.join(dirpath, filename)
        cv2.imwrite(output_filepath, img)
        
    def saveSegmentedImages(self):
        """
        Save 3 kinds of images.
        1. input image which detected text regions  are drawn
        2. mask image by each text region
        3. inpainted image by each generated mask image

        """
        if not os.path.isdir(self.output_for_charRecognition_dirpath):
            os.makedirs(self.output_for_charRecognition_dirpath)

        filename = os.path.basename(self.img_filepath)
        stem, ext = os.path.splitext(filename)

        #self.saveImage(self.img, filename)
        output_filename = '{}.jpg'.format(stem)
        
        # all array has only one elements, so unnesessary to name the file
        for i, text_region_img in enumerate(self.text_region_imgs):
            output_filename = '{}_{:02d}.jpg'.format(stem, i)
            output_dirpath = os.path.join(self.output_for_learning_dirpath, "title")
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

            self.saveImage(text_region_img, output_dirpath, output_filename)
        
        for i, masked_img in enumerate(self.masked_imgs):
            output_filename = '{}_{:02d}.jpg'.format(stem, i)

             # for making datase
            output_dirpath = os.path.join(self.output_for_learning_dirpath, "cover_mask")
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

            self.saveImage(masked_img, output_dirpath, output_filename)
    
        for i, pts in enumerate(self.text_region_pts):
            output_filename = '{}_{:02d}.txt'.format(stem, i)

             # for making datase
            output_dirpath = os.path.join(self.output_for_learning_dirpath, "text_region")
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

            with open(os.path.join(output_dirpath, output_filename), mode="w") as f:
                for ele in pts:
                    f.write(str(ele[0])+','+str(ele[1])+'\n')


        # for i, inpainted_img in enumerate(self.inpainted_imgs):
        #     output_filename = '{}_{:02d}.jpg'.format(stem, i)
        #      # for making datase
        #     output_dirpath = os.path.join(self.output_for_learning_dirpath, "cover_inpaint")
        #     if not os.path.exists(output_dirpath):
        #         os.makedirs(output_dirpath)

        #     self.saveImage(inpainted_img, output_dirpath, output_filename)
        
        output_filename = '{}.jpg'.format(stem)
        output_dirpath = os.path.join(self.output_for_learning_dirpath, "cover_inpaint")
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        self.saveImage(self.all_pts_inpainted_img, output_dirpath, output_filename)



    def detectAndSegmentChars(self, output_for_learning_dirpath=None,output_for_charRecognition_dirpath=None):
        """
        Main function of Segmentation.
        """
        self.deleteFlag = False

        #1
        self.segmentTextRegions()
        #2
        self.inpaintingAtMaskeRegions()
        #3
        #self.drawTextRegions()
        #self.segmentChars(labeling=True)
        if self.deleteFlag == True:
            # if the image is unsuitable for training, delete the image and CRAFT text data
            os.remove(self.img_filepath)
            os.remove(self.txt_filepath)

        elif output_for_learning_dirpath and len(self.text_region_imgs) > 0:
            self.output_for_learning_dirpath = output_for_learning_dirpath
            self.output_for_charRecognition_dirpath = output_for_charRecognition_dirpath
            self.saveSegmentedImages()


def main(args):
    """
    Main function.
    """
    if os.path.isdir(args.img_path) and os.path.isdir(args.txt_path):
        from tqdm import trange
      
        img_filepaths = []
        txt_filepaths = []
        for filename in os.listdir(args.img_path):
            stem, ext = os.path.splitext(filename)
            txt_filepath = os.path.join(args.txt_path, stem + '.txt')
            if os.path.isfile(txt_filepath):
                txt_filepaths.append(txt_filepath)
                img_filepaths.append(os.path.join(args.img_path, filename))
        print('[Segment text regions from {} images]'.format(len(img_filepaths)))
    
    else:
        
        img_filename = os.path.basename(args.img_path)

        txt_filename = os.path.basename(args.txt_path)

        img_stem, img_ext = os.path.splitext(img_filename)
        txt_stem, txt_ext = os.path.splitext(txt_filename)
        if img_stem == txt_stem and img_ext in {'.jpg', '.png'} and txt_ext == '.txt':
            x = Segmentation(args.img_path, args.txt_path)
            x.detectAndSegmentChars(output_for_learning_dirpath=args.output_for_learning_dirpath,output_for_charRecognition_dirpath=args.output_for_charRecognition_dirpath)
            print(
                '.. Found {} text region candidates '.format(len(x.text_region_imgs)
                                                            + len(x.non_text_region_pts))
                + '(use {} candidates)\n'.format(len(x.text_region_imgs))
                + '.. Found {} character candidates'.format(x.char_imgs.shape[0])
            )


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    
    parser = ArgumentParser()
    parser.add_argument(
        'img_path',
        type=str,
        help='test image filepath or path of directory including test images'
    )
    parser.add_argument(
        'txt_path',
        type=str,
        help=('path of .txt file in which is written the coordinates of detected text region'
              + 'bounding boxes or path of directory which includes them')
    )
    parser.add_argument(
        'output_for_learning_dirpath',
        type=str,
        help='output directory path of the images for machine learning(mask and inpainting)'
    )
    parser.add_argument(
        'output_for_charRecognition_dirpath',
        type=str,
        help='output directory path of segmented character images'
    )
    args = parser.parse_args()
    
    main(args)
