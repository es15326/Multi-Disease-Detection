# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-resnext-fixes/checkpoints
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-good-models/single
# MODELS_DIR=/data/code/retinal-disease-classification/weights-resnextbugfix-final/checkpoints/best
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-resnextbugfix-es-loss-meru-inv-freq-real-224/checkpoints/best
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-clip-l224-nodinit-update-norm/checkpoints
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-clip-l224-noinit-noupdate-norm/checkpoints/best
# MODELS_DIR=/cluster/VAST/civalab/results/riadd/imad/final-resnext-512-noinit-noupdate-norm-newsplit/best
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-crossval-resnext512-noinit-noupdate-norm/best
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-resnext512-noinit-noupdate-norm-noother/checkpoints/best
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-resnext512-init-update-itdfhnorm/checkpoints/best

# MODELS_DIR=/cluster/VAST/civalab/results/riadd/imad/final-resnext-512-noinit-noupdate-norm-newsplit/best
# MODELS_DIR=/cluster/VAST/civalab/results/riadd/imad/final-resnext-512-init-noupdate-itdfhnorm-nautilus/checkpoints
# MODELS_DIR=/cluster/VAST/civalab/results/riadd/imad/final-resnext-512-noinit-update-itdfhnorm-nautilus/checkpoints
# MODELS_DIR=/cluster/VAST/civalab/results/riadd/imad/final-resnext-512-noinit-update-itdfhnorm/checkpoints

#ARCH=resnext
#ARCH=vitb8
ARCH=vitb16

# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-network-ablation-512-ablation-architectures/checkpoints/$ARCH
# MODELS_DIR=/usr/mvl2/itdfh/dev/retinal-disease-classification/weights-resnext512-init-update-itdfhnorm/checkpoints/best
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-dino224-init-update-itdfhnorm_v2/weights-dino224-init-update-itdfhnorm/checkpoints/best
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-dino512-init-update-itdfhnorm/checkpoints/best
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-dino768-init-update-itdfhnorm/checkpoints/best
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-vitb8224-init-update-itdfhnorm/checkpoints/best3
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/best_ckpt
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/best_ckpt

# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-vitb8224-init-update-itdfhnorm/checkpoints/best
# MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-vitb8224-init-update-itdfhnorm/checkpoints/best
MODELS_DIR=/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin_github/Multi-Disease-Detection/models/checkpoints/

BASEDIR=/usr/mvl2/esdft/ISBI_elham

VAL_IMG_PATH=$BASEDIR
VAL_CSV=riadd_new_eval.csv

TEST_IMG_PATH=$BASEDIR/Test_Set/Test/
TEST_CSV=$BASEDIR/Test_Set/RFMiD_Testing_Labels.csv


# TEST_IMG_PATH=$BASEDIR/Evaluation_Set/Validation/
# TEST_CSV=$BASEDIR/Evaluation_Set/RFMiD_Validation_Labels.csv



# TEST_IMG_PATH=$BASEDIR/Training_Set/Training/
# TEST_CSV=$BASEDIR/Training_Set/RFMiD_Training_Labels.csv



AGGREGATION=average
RESOLUTION=224

# python eval.py \
#   --models-dir $MODELS_DIR --test-img-path $VAL_IMG_PATH --test-csv $VAL_CSV \
#   --arch $ARCH --aggregation $AGGREGATION --input-resolution $RESOLUTION \
#   --out_file val_outs.npy
python eval.py \
  --models-dir $MODELS_DIR --test-img-path $TEST_IMG_PATH --test-csv $TEST_CSV \
  --arch $ARCH --aggregation $AGGREGATION --input-resolution $RESOLUTION \
  --out_file outs_train_data.npy
