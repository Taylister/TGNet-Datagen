import os
import sys
import argparse
import imghdr
import random
import shutil
import tqdm


# the structure of data_dir is like below
# dataset
# └ cover_inpaint
# └ cover_mask
# └ extracted_title
# └ input_text
# └ processed_mask
# └ skeletonize
# └ resized_title
#
#  this program copy the files like a below structure
# dataset
# └ train
#    └ cover_inpaint
#    └ cover_mask
#    └ extracted_title
#    └ input_text
#    └ processed_mask
#    └ skeletonize
#    └ resized_title
# └ test
#    └ cover_inpaint
#    └ cover_mask
#    └ extracted_title
#    └ input_text
#    └ processed_mask
#    └ skeletonize
#    └ resized_title
# └ validation
#    └ cover_inpaint
#    └ cover_mask
#    └ extracted_title
#    └ input_text
#    └ processed_mask
#    └ skeletonize
#    └ resized_title


def main(args):
    data_dir = os.path.expanduser(args.data_dir)
    
    print('loading dataset')

    filename_list = []
    target_dir_path = os.path.join(data_dir,"cover_mask")

    for file in os.listdir(target_dir_path):
        path = os.path.join(target_dir_path, file)
        if imghdr.what(path) == None:
            continue

        filename_list.append(file)
    
    book_IDs = []
    print("# Making Book Cover ID list")
    for filename in filename_list:
        #filename = "0761525696_01.jpg"
        filename_without_ext, ext = os.path.splitext(os.path.basename(filename))
        #filename_without_ext = "0761525696_01" ,ext = ".jpg"
        book_ID = filename_without_ext.rsplit("_",1)[0]
        book_IDs.append(book_ID)
    
    # Create idependate list
    book_IDs = list(set(book_IDs))

    random.shuffle(book_IDs)

    # separate the paths
    test_border = int(0.1 * len(book_IDs))
    if 2000 <= test_border:
        test_border = 2000

    test_cover_IDs = book_IDs[:test_border]
    train_cover_IDs = book_IDs[test_border:]

    print('the number of test covers: %d' % len(test_cover_IDs))
    print('the number of train covers: %d' % len(train_cover_IDs))

    # create dst directories
    print(data_dir)
    data_dirs = os.listdir(data_dir)
    pbar = tqdm.tqdm(len(data_dirs),total=len(data_dirs))
    exclude = ['raw_mask','title','original.csv','processed.csv']
    for dir in data_dirs:
        if dir in exclude:
            continue
        
        test_dir = os.path.join(data_dir,dir,"test")
        train_dir = os.path.join(data_dir,dir,"train")
        
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        
        for book_ID in test_cover_IDs:
            cover_images = []
            cover_images = [file for file in filename_list if book_ID in file]
            for cover_image in cover_images:
                src_image_path = os.path.join(data_dir,dir,cover_image)
                dst_image_path = os.path.join(data_dir,dir,"test",cover_image)
                try:
                    shutil.move(src_image_path, dst_image_path)
                except Exception as e:
                    pass

        for book_ID in train_cover_IDs:
            cover_images = []
            cover_images = [file for file in filename_list if book_ID in file]
            for cover_image in cover_images:
                src_image_path = os.path.join(data_dir,dir,cover_image)
                dst_image_path = os.path.join(data_dir,dir,"train",cover_image)
                try:
                    shutil.move(src_image_path, dst_image_path)
                except Exception as e:
                    pass
        
        surplus_file_list = os.listdir(os.path.join(data_dir,dir))
        for name in surplus_file_list:
            filepath = os.path.join(data_dir,dir,name)
            if os.path.isfile(filepath):
                os.remove(filepath)

        pbar.update(1)
        
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args()
    main(args)