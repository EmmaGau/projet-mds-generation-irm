from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    diffusion_defaults
)
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import tqdm
import torch
from finetune_unet import FinetuneUNetModel
import argparse
from dataloader import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        out_channels=3,
    )
    res.update(diffusion_defaults())
    return res

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    out_channels=3
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return FinetuneUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    #parser.add_argument('--out_channels', type=int, default=3)
    return parser


parser = create_argparser()
parser.add_argument('model_path')
parser.add_argument('img_dir')
parser.add_argument('result_dir')
args = parser.parse_args()
for k, v in args.__dict__.items():
    print(k, v)

dist_util.setup_dist()
logger.configure()
model = create_model(
    args.image_size,
    args.num_channels,
    args.num_res_blocks,
    args.channel_mult,
    False,
    args.class_cond,
    args.use_checkpoint,
    args.attention_resolutions,
    args.num_heads,
    args.num_head_channels,
    args.num_heads_upsample,
    args.use_scale_shift_norm,
    args.dropout,
    args.resblock_updown,
    args.use_fp16,
    args.use_new_attention_order,
    args.out_channels
)

categories = {
'Email' : ([255,0,0], 1),
'Os' : ([0,255,0], 2),
'Dentine': ([0,0,255], 3),
'Autre' : ([255, 152, 0], 4),
'Carie': ([255,152,0], 5),
'Pulpe': ([0, 255, 237], 6)
}

colors = {v[1] : v[0] for k, v in categories.items()}

state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)

device = 'cuda'

model = model.to(device)

for name, param in model.named_parameters():
    param.requires_grad = False

def overlay_images(image, overlay, ignore_color=[0, 0, 0], alpha=0.5):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay == ignore_color).all(-1, keepdims=True)
    out = np.where(mask, image, image * alpha + overlay * (1 - alpha)).astype(image.dtype)
    return out

# Create output directory if needed
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

model.eval()
torch.backends.cudnn.deterministic = True
for i in os.listdir(args.img_dir):
    img = cv2.imread(os.path.join(args.img_dir, i))
    height, width = img.shape[:2]

    img = cv2.resize(img, (args.image_size, args.image_size))
    orig_img = img.copy()
    img = img.astype(np.float32) / 127.5 - 1.
    img = np.moveaxis(img, -1, 0)
    batch = np.array([img])
    tensor = torch.from_numpy(batch)
    tensor = tensor.to(device, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor, timesteps = torch.full((tensor.size()[0],), 1, device='cuda', dtype=torch.int32))
        outputs = torch.softmax(outputs[0], axis=0)
        tmp = outputs.detach().cpu().numpy()
        tmp = np.argmax(tmp, axis=0)
        res = np.zeros_like(img, dtype=np.uint8).transpose(1,2,0)

        for c_number, c_color in colors.items():
            res[tmp == c_number] = c_color

        # fig, axes = plt.subplots(nrows=1, ncols=3)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        orig_img_cpy = orig_img.copy()
        res_cpy = res.copy()
        # axes[0].imshow(orig_img)
        # axes[1].imshow(res)
        # axes[2].imshow(overlay_images(orig_img_cpy, res_cpy,alpha=0.85))
        # plt.show()

        cv2.imwrite(os.path.join(args.result_dir, i.split('.')[0] + '_orig.' + i.split('.')[1]), orig_img)
        cv2.imwrite(os.path.join(args.result_dir, i.split('.')[0] + '_res.' + i.split('.')[1]), res)
        cv2.imwrite(os.path.join(args.result_dir, i.split('.')[0] + '_overlay.' + i.split('.')[1]), overlay_images(orig_img_cpy, res_cpy,alpha=0.85))
 