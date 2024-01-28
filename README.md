# projet-mds-generation-irm

## Clone guided-diffusion

https://github.com/openai/guided-diffusion in the folder guided_diffusion

```git clone https://github.com/openai/guided-diffusion guided_diffusion```

## Scripts

The command lines to run the python scripts being quite long it's easier to store them in bash scripts (you might need to change them a bit, please refer to the guide-diffusion documentation to understand the commands in details)

- `./unet_paper.sh` to train the UNet
- `./unet_finetune.sh` to finetune the model
- `./unet_predict.sh` to generate segmentation