MODEL_FLAGS='--num_channels 128 --num_res_blocks 2 --attention_resolutions 32,16,8 --num_head_channels 64 --resblock_updown True --use_scale_shift True'
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --microbatch 1"

# python predict.py checkpoints/finetune/segm200_256.pth data/Dataset_segmentation/Split/VAL/IMG data/Results/Split/VAL/ $MODEL_FLAGS $TRAIN_FLAGS --lr 1e-4 --image_size 256  --out_channels 7 --batch_size 2

python predict.py checkpoints/finetune/segm200_64_split3.pth data/Dataset_segmentation/Split3/VAL/IMG data/Results/Split3/VAL/ $MODEL_FLAGS $TRAIN_FLAGS --lr 1e-4 --image_size 64  --out_channels 7 --batch_size 2
python predict.py checkpoints/finetune/segm200_64_split3.pth data/Dataset_segmentation/Split3/TEST/IMG data/Results/Split3/TEST/ $MODEL_FLAGS $TRAIN_FLAGS --lr 1e-4 --image_size 64  --out_channels 7 --batch_size 2


# Results: Epoch 200 Val loss: 0.021 mIoU 56.222% elapsed time 7.3s
# Results: Epoch 107 Val loss: 0.030 mIoU 46.091% elapsed time 6.6s