#!/bin/bash

BASE_LOGDIR="checkpoints"
MODEL_FLAGS='--num_channels 128 --num_res_blocks 2 --attention_resolutions 32,16,8 --num_head_channels 64 --resblock_updown True --use_scale_shift True'
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"

# Format the current date and time
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# Create the directory with the date
OPENAI_LOGDIR="$BASE_LOGDIR/$TIMESTAMP"

# Run the training script
OPENAI_LOGDIR=$OPENAI_LOGDIR python image_train.py --data_dir data/Bitewings_Resized_256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
