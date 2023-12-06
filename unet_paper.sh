OPENAI_LOGDIR="checkpoints_unet_truepaper"
MODEL_FLAGS='--num_channels 128 --num_res_blocks 2 --attention_resolutions 32,16,8 --num_head_channels 64 --resblock_updown True --use_scale_shift True'
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --microbatch 1"

SAMPLE_FLAGS="--num_samples 32 --timestep_respacing 250"
