from PIL import Image
from datasets import Dataset
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from utils import TrainingConfig
import pandas as pd

config = TrainingConfig()

# Supposons que vous avez une liste de noms de fichiers d'images
data_dir = "../data/Bitewings_Resized_256/"
image_files = os.listdir(data_dir)  # Remplacez ces noms par les noms de vos fichiers

# Charger les images et les stocker dans une liste
images = []
for image_file in image_files:
    with Image.open(data_dir + image_file) as img:
        images.append(img)

# Création du dataset avec les images chargées
custom_dataset = Dataset.from_dict({"image": images})


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


custom_dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=config.train_batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
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

model = model.to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    df_log = pd.DataFrame(columns=['epoch', 'loss', 'global_step', 'lr'])
    df_log.to_csv(config.output_dir + '/log.csv', index=False, columns=['epoch', 'loss', 'global_step', 'lr'])

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        total_loss = 0

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().item()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        new_log = pd.DataFrame({'epoch': epoch, 'loss': total_loss, 'global_step': global_step, 'lr': lr_scheduler.get_last_lr()[0]}, index=[0])
        if len(df_log) > 0:
            df_log = pd.concat([df_log, new_log], ignore_index=True)
        else:
            df_log = new_log
        df_log.to_csv(config.output_dir + '/log.csv', index=False, columns=['epoch', 'loss', 'global_step', 'lr'])

        # plot loss
        fig = plt.figure()
        plt.plot(df_log['epoch'], df_log['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.savefig(config.output_dir + '/loss.png')


        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                print(f"Sampling images at epoch {epoch}...")
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                print(f"Saving model at epoch {epoch}...")
                pipeline.save_pretrained(config.output_dir)

if __name__ == "__main__":
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)