#!/bin/bash

BASEDIR=/usr/mvl2/esdft/ISBI_elham

TRAIN_IMG_PATH=$BASEDIR
VAL_IMG_PATH=$BASEDIR
# TRAIN_CSV=$BASEDIR/Training_Set/RFMiD_Training_Labels.csv
# VAL_CSV=$BASEDIR/Evaluation_Set/RFMiD_Validation_Labels.csv
TRAIN_CSV=riadd_new_train.csv
VAL_CSV=riadd_new_eval.csv

RESOLUTION=224
BATCH_SIZE=8
BOOSTING_EXPERTS=1
BOOSTING_RESUME_FROM=0
MAX_EPOCHS=200
TAG=$RESOLUTION-init-update-itdfhnorm

# ARCH=resnext # options: resnext, vit-l, vit-b, vit-h, clip-l
ARCH=vitb8


# echo ' \
#   "--train_img_path", "'"${TRAIN_IMG_PATH}"'",
#   "--val_img_path", "'"${VAL_IMG_PATH}"'",
#   "--train_csv", "'"${TRAIN_CSV}"'",
#   "--val_csv", "'"${VAL_CSV}"'",
#   "--resolution", "'"${RESOLUTION}"'",
#   "--batch_size", "'"${BATCH_SIZE}"'",
#   "--boosting_experts", "'"${BOOSTING_EXPERTS}"'",
#   "--boosting_resume_from", "'"${BOOSTING_RESUME_FROM}"'",
#   "--max_epochs", "'"${MAX_EPOCHS}"'",
#   "--arch", "'"${ARCH}"'",
#   "--tag", "'"${TAG}"'"'
CUDA_VISIBLE_DEVICES=2 python train.py \
  --train_img_path $TRAIN_IMG_PATH \
  --val_img_path $VAL_IMG_PATH \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --resolution $RESOLUTION \
  --batch_size $BATCH_SIZE \
  --boosting_experts $BOOSTING_EXPERTS \
  --boosting_resume_from $BOOSTING_RESUME_FROM \
  --max_epochs $MAX_EPOCHS \
  --arch $ARCH \
  --tag $TAG
