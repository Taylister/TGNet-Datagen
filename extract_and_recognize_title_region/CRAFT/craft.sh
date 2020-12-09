TRAIN_TARGET_DIRPATH="/Users/taiga/Desktop/卒業研究/Code/Main/dataset/AmazonCover/train/real/"
OUTPUT_DIRPATH="/Users/taiga/Desktop/卒業研究/Code/Main/result/TextSegmentation/BoundingBoxInfo/"


python test.py \
        --trained_model="weight/craft_mlt_25k.pth" \
        --link_threshold=1000 \
        --test_folder=${TRAIN_TARGET_DIRPATH} \
        --output_dir_path=${OUTPUT_DIRPATH}
        
# in order to segment each character area, setting link_threshhold to 1000