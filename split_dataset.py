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
    target_dir_path = os.path.join(data_dir,"cover_inpaint")

    for file in os.listdir(target_dir_path):
        path = os.path.join(target_dir_path, file)
        if imghdr.what(path) == None:
            continue

        filename_list.append(file)

    random.shuffle(filename_list)

    # separate the paths
    test_border = int(0.1 * len(filename_list))
    if 1000 <= test_border:
        test_border = 1000
    val_border = int(0.3 * len(filename_list))

    test_image_name = filename_list[:test_border]
    val_image_name = filename_list[test_border:val_border]
    train_image_name = filename_list[val_border:]

    print('the number of test images: %d' % len(test_image_name))
    print('the number of validation images: %d' % len(val_image_name))
    print('the number of train images: %d' % len(train_image_name))

    # create dst directories
    print(data_dir)
    data_dirs = os.listdir(data_dir)
    pbar = tqdm.tqdm(len(data_dirs),total=len(data_dirs))
    exclude = ['raw_mask','title','original.csv','processed.csv']
    for dir in data_dirs:
        if dir in exclude:
            continue
        
        test_dir = os.path.join(data_dir,dir,"test")
        validation_dir = os.path.join(data_dir,dir,"validation")
        train_dir = os.path.join(data_dir,dir,"train")
        
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
        
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        for filename in test_image_name:
            src_image_path = os.path.join(data_dir,dir,filename)
            dst_image_path = os.path.join(data_dir,dir,"test",filename)
            shutil.move(src_image_path, dst_image_path)

        for filename in val_image_name:
            src_image_path = os.path.join(data_dir,dir,filename)
            dst_image_path = os.path.join(data_dir,dir,"validation",filename)
            shutil.move(src_image_path, dst_image_path)

        for filename in train_image_name:
            src_image_path = os.path.join(data_dir,dir,filename)
            dst_image_path = os.path.join(data_dir,dir,"train",filename)
            shutil.move(src_image_path, dst_image_path)

        pbar.update(1)
        
    pbar.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')

    args = parser.parse_args()
    main(args)