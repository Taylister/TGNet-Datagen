#!/usr/bin/env bash
set -eu

############################################################################################
# Setting path
############################################################################################

echo "setting up dirs ($(date))"

main_dir=$(pwd)'/'
echo "This is $main_dir"

TARGET_DIRPATH="${main_dir}/test_"

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
          --image_folder "${SEGMENTATION_RESULT_DIRPATH}" \
          --output_dirpath "${OUTPUT_DIRPATH}" \
          --sensitive \
          --saved_model "${TEXT_RECOGNITION_DIRPATH}/weight/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

  cd ../../

fi

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

############################################################################################
# Generate mask of character region of title part image  
############################################################################################

 python3 image_processing.py \
          "${OUTPUT_DIRPATH}/raw_mask" \
          "${OUTPUT_DIRPATH}/title" \
          "${OUTPUT_DIRPATH}"