#!/usr/bin/env bash
set -eu

############################################################################################
# Setting path
############################################################################################

echo "setting up dirs ($(date))"

main_dir=$(pwd)'/'
echo "This is $main_dir"

TARGET_DIRPATH="${main_dir}/test_"
SEGMENTATION_RESULT_DIRPATH="${main_dir}/TitleSegmentation/"
CRAFT_RESULT_DIRPATH="${SEGMENTATION_RESULT_DIRPATH}/BoundingBoxInfo"
OUTPUT_DIRPATH="${main_dir}/result"

############################################################################################
# Extract text region as bounding box
############################################################################################

if [ -e ${CRAFT_RESULT_DIRPATH} ]; then
  echo "Tittle region info is already exist."
else
  mkdir -p ${CRAFT_RESULT_DIRPATH}
  echo "Strat Extract title region."
  cd CRAFT
  python TextDetection.py \
       --trained_model="weight/craft_mlt_25k.pth" \
       --test_folder="${TARGET_DIRPATH}" \
       --output_dir_path="${CRAFT_RESULT_DIRPATH}"
  cd ..

fi

echo "----------"

# in order to segment each character area, setting link_threshhold to 1000

############################################################################################
# Save extracted region image as origin and mask format
############################################################################################

python System.py \
       ${OUTPUT_DIRPATH} \
       --book_img_dirpath=${TARGET_DIRPATH} \
       --craft_result_dirpath=${CRAFT_RESULT_DIRPATH} \
       --segmented_img_dirpath=${SEGMENTATION_RESULT_DIRPATH} \
       #--segmented_img_dirpath=${SEGMENTATION_RESULT_DIRPATH} \
cd text-recognition

echo "----------"

############################################################################################
# 
############################################################################################

python3 TextRecognition.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--book_img_dirpath "${TARGET_DIRPATH}" \
--image_folder "${SEGMENTATION_RESULT_DIRPATH}" \
--output_dirpath "${OUTPUT_DIRPATH}" \
--sensitive \
--saved_model weight/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth

cd ..

