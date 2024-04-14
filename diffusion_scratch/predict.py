import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from diffusers import UNet2DModel


parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
parser.add_argument('--img_dir')
parser.add_argument('--result_dir')
parser.add_argument('--out_channels', type=int, default=7)
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

model = UNet2DModel(
            sample_size=args.image_size,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=args.out_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

model.load_state_dict(torch.load(args.model_path))

categories = {
'Email' : ([255,0,0], 1),
'Os' : ([0,255,0], 2),
'Dentine': ([0,0,255], 3),
'Autre' : ([255, 0, 254], 4),
'Carie': ([255,152,0], 5),
'Pulpe': ([0, 255, 237], 6)
}

colors = {v[1] : v[0] for k, v in categories.items()}

device = ('cuda' if torch.cuda.is_available() else 'cpu')

state_dict = torch.load(args.model_path, map_location=device)
model.load_state_dict(state_dict)

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

out_dir = ['IMG/', 'MASK/', 'OVERLAY/']
for i in out_dir:
    print(os.path.join(args.result_dir, i))
    if not os.path.exists(os.path.join(args.result_dir, i)):
        os.makedirs(os.path.join(args.result_dir, i))

model.eval()
torch.backends.cudnn.deterministic = True
for i in os.listdir(args.img_dir):
    img = cv2.imread(os.path.join(args.img_dir, i), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (args.image_size, args.image_size))
    batch = np.array([[img]])
    tensor = torch.from_numpy(batch)
    tensor = tensor.to(device, dtype=torch.float32)
    orig_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    with torch.no_grad():
        outputs = model(tensor, torch.full((tensor.size()[0],), 1, device=device, dtype=torch.int32))[0]
        outputs = torch.softmax(outputs[0], axis=0)
        tmp = outputs.detach().cpu().numpy()
        print(tmp.shape)
        tmp = np.argmax(tmp, axis=0)
        print(tmp.shape)
        print(np.unique(tmp))
        res = np.zeros((tmp.shape[0], tmp.shape[1], 3), dtype=np.uint8)

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
        
        print('writing results for', i)
        cv2.imwrite(os.path.join(args.result_dir, 'IMG/', i), orig_img)
        cv2.imwrite(os.path.join(args.result_dir, 'MASK/', i), res)
        cv2.imwrite(os.path.join(args.result_dir, 'OVERLAY/', i), overlay_images(orig_img_cpy, res_cpy,alpha=0.85))
 