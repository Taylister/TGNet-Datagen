#!/usr/bin/env bash
set -eu

# Segmentation.py中のself.all_pts_inpainted_imgを保存しない場合は、
# 全ての文字領域をinpaintされた画像は保存されない

############################################################################################
# Setting path
############################################################################################

echo "setting up dirs ($(date))"

main_dir=$(pwd)'/'
echo "This is $main_dir"

TARGET_DIRPATH="${main_dir}/cover"
#TARGET_DIRPATH="${main_dir}/test_"

# each module path
EXTRACT_AND_RECOGNIZE_TITLE_REGION_DIRPATH="${main_dir}extract_and_recognize_title_region" 
CRAFT_DIRPATH="${EXTRACT_AND_RECOGNIZE_TITLE_REGION_DIRPATH}/CRAFT"
TEXT_RECOGNITION_DIRPATH="${EXTRACT_AND_RECOGNIZE_TITLE_REGION_DIRPATH}/text-recognition"
CSNet_DIRPATH="${main_dir}CSNet"

# output dirpath
SEGMENTATION_RESULT_DIRPATH="${EXTRACT_AND_RECOGNIZE_TITLE_REGION_DIRPATH}/TitleSegmentation"
CRAFT_RESULT_DIRPATH="${SEGMENTATION_RESULT_DIRPATH}/BoundingBoxInfo"

OUTPUT_DIRPATH="${main_dir}/tg-data"

############################################################################################
# Extract text region as bounding box
############################################################################################

if [ -e ${CRAFT_RESULT_DIRPATH} ]; then
  echo "Title region info is already exist."
else
  mkdir -p ${CRAFT_RESULT_DIRPATH}
  echo "Strat extracting title region."
  cd extract_and_recognize_title_region/CRAFT
  python TextDetection.py \
       --trained_model="${CRAFT_DIRPATH}/weight/craft_mlt_25k.pth" \
       --test_folder="${TARGET_DIRPATH}" \
       --cuda="yes"\
       --output_dir_path="${CRAFT_RESULT_DIRPATH}"
  cd ../../

fi

echo "----------"

# in order to segment each character area, setting link_threshhold to 1000

############################################################################################
# Save extracted region image
############################################################################################

if [ -e ${OUTPUT_DIRPATH}/title ]; then
  echo "Title region image is already exist."
else
  cd extract_and_recognize_title_region
  echo "Strat segment and save title region."
  python System.py \
         ${OUTPUT_DIRPATH} \
         --book_img_dirpath=${TARGET_DIRPATH} \
         --craft_result_dirpath=${CRAFT_RESULT_DIRPATH} \
         --segmented_img_dirpath=${SEGMENTATION_RESULT_DIRPATH} \

  cd ..

fi

echo "----------"

############################################################################################
# Save text information from title part image(original title part)
############################################################################################


if [ -e ${OUTPUT_DIRPATH}/original.csv ]; then
  echo "Recognized title information is already exist."
else
  echo "Start recognizing text in title image "
  cd extract_and_recognize_title_region/text-recognition

  python3 TextRecognition.py \
          --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
          --book_img_dirpath "${TARGET_DIRPATH}" \
          --csv_file_name "original.csv" \
          --image_folder "${OUTPUT_DIRPATH}/title" \
          --output_dirpath "${OUTPUT_DIRPATH}" \
          --sensitive \
          --saved_model "${TEXT_RECOGNITION_DIRPATH}/weight/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

  cd ../../

fi

echo "----------"

############################################################################################
# Generate mask of character region of title part image  
############################################################################################

if [ -e ${OUTPUT_DIRPATH}/raw_mask ]; then
  echo "Mask of character region of the title part images are already exist."
else
  echo "Start generating mask of character region of title part image "
  cd CSNet

  python3 predict.py \
          "${OUTPUT_DIRPATH}/title" \
          "${OUTPUT_DIRPATH}/raw_mask" \
          "${CSNet_DIRPATH}/weight/CSNet_weight.pth"

  cd ..

fi

echo "----------"

############################################################################################
# Generate mask of character region of title part image  
############################################################################################

if [ -e ${OUTPUT_DIRPATH}/extracted_title ]; then
  echo "Image processing to make datasets is already completed"
else
  python3 image_processing.py \
          "${OUTPUT_DIRPATH}/raw_mask" \
          "${OUTPUT_DIRPATH}/title" \
          "${main_dir}/font/arial.ttf" \
          "${OUTPUT_DIRPATH}/original.csv" \
          "${OUTPUT_DIRPATH}"
fi

echo "----------"

############################################################################################
# Save text information from title part image(processed title part)
############################################################################################

if [ -e ${OUTPUT_DIRPATH}/processed.csv ]; then
  echo "Recognized title information is already exist."
else
  echo "Start recognizing text in title image "
  cd extract_and_recognize_title_region/text-recognition

  python3 TextRecognition.py \
          --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
          --book_img_dirpath "${TARGET_DIRPATH}" \
          --csv_file_name "processed.csv" \
          --image_folder "${OUTPUT_DIRPATH}/extracted_title" \
          --output_dirpath "${OUTPUT_DIRPATH}" \
          --sensitive \
          --saved_model "${TEXT_RECOGNITION_DIRPATH}/weight/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

  cd ../../

fi

###########################################################################################
# Save text information from title part image(processed title part)
###########################################################################################

python3 select_data.py \
        "${OUTPUT_DIRPATH}" \
        "${OUTPUT_DIRPATH}/original.csv" \
        "${OUTPUT_DIRPATH}/processed.csv"

if [ -e ${main_dir}/sample_dataset ]; then
  echo "TGNet-Dataset is already exist."
else
  echo "Start configurating the data"

  python3 make_dataset.py \
        "${OUTPUT_DIRPATH}" \
        "${main_dir}/dataset" \

fi