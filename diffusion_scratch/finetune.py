from utils import TrainingConfig
import torch
from dataloader import DatasetSeg
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
import cv2
import os
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import time
from tqdm import tqdm


def main():
    config = TrainingConfig()

    img_train = '../data/Dataset_segmentation/Split3/TRAIN/IMG'
    mask_train = '../data/Dataset_segmentation/Split3/TRAIN/MASK'
    img_val = '../data/Dataset_segmentation/Split3/VAL/IMG'
    mask_val = '../data/Dataset_segmentation/Split3/VAL/MASK'
    pretrained_model = 'ddpm-bitewings-64/unet'
    class_wheighting = False
    config.image_size = 64
    args= {'class_weighting': False, 'lr': 1e-4, 'epochs':5, 'out_channels': 7}

    # Charger les images et les stocker dans une liste
    dataset_train = DatasetSeg(img_train, mask_train, config.image_size)
    training_generator = torch.utils.data.DataLoader(dataset_train, batch_size=config.train_batch_size, shuffle=True)
    dataset_val = DatasetSeg(img_val, mask_val, config.image_size)
    validation_generator = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    # check the first element of the training set
    img, mask = next(iter(training_generator))
    print(config.image_size)
    print(img.size(), mask.size())
    print(torch.unique(mask))

    model_pretrained = UNet2DModel(
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

    model_pretrained.from_pretrained(pretrained_model)

    model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=6,  # the number of output channels
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

    state_dict = model_pretrained.state_dict()
    del state_dict['conv_out.weight']
    del state_dict['conv_out.bias']
    model.load_state_dict(state_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    val_losses = []
    val_mious = []
    all_lr = []
    best_loss = None
    best_miou = None

    # Class weighting
    if class_wheighting:
        print('Calculating class weights ...')
        histo = {i: 0 for i in range(7)}
        for mask_path in os.listdir(mask_train):
            m = cv2.imread(os.path.join(mask_train, mask_path), -1)
            unique_values, unique_counts = np.unique(np.array(m).flatten(), return_counts=True)
            for val, c in zip(unique_values, unique_counts):
                histo[val] += c

        weights = []
        total = np.array(list(histo.values())).sum()
        for i in range(7):
            weights.append(1 - histo[i] / total)
        weights = torch.FloatTensor(weights).to(device)
        print('Done ! ', weights)

    criterion = torch.nn.CrossEntropyLoss(weight=weights if args['class_weighting'] else None)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args['epochs'])
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=args['out_channels']).to(device)

    for epoch in range(args['epochs']):  # loop over the dataset multiple times
            model.train()
            running_loss = []
            start = time.time()

            all_lr.append(scheduler.state_dict()['_last_lr'])
            for img, mask in tqdm(training_generator, total=len(training_generator)):
                img, mask = img.to(device), mask.to(device)
                # print(img.size(), img.min(), img.max(), img.type(), mask.size(), mask.min(), mask.max(), mask.type())
                # zero the parameter gradients
                optimizer.zero_grad()

                # On donne l'image d'origine avec un timestep fixe à 1 (pour simuler une image quasiment pas bruitée pour
                # rester cohérent avec le pretrain)
                outputs = model(img, torch.tensor([1.], dtype=torch.float32, device=device))
                print(outputs.size(), mask.size())

                loss = criterion(outputs, mask.long())

                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())

            running_loss = np.array(running_loss).sum() / len(training_generator)
            end = time.time()
            print('Epoch %d loss: %.3f elapsed time %.1fs' % (epoch + 1, running_loss, end - start))
            train_losses.append(running_loss)

            if validation_generator is not None:
                model.eval()
                start = time.time()
                val_loss = []
                mious = []
                with torch.no_grad():
                    for img, mask in tqdm.tqdm(validation_generator, total=len(validation_generator)):
                        img, mask = img.to(device), mask.to(device)
                        outputs = model(img, torch.tensor([1.], dtype=torch.float32, device=device))
                        loss = criterion(outputs, mask.long())
                        mious.append(jaccard(outputs, mask).item())
                        val_loss.append(loss.item())
                val_loss = np.array(val_loss)
                end = time.time()
                new_loss = val_loss.mean() / len(validation_generator)
                new_miou = np.array(mious).mean()
                print("Epoch %d Val loss: %.3f mIoU %.3f%% elapsed time %.1fs" % (
                    epoch + 1, new_loss, new_miou * 100, end - start))
                val_losses.append(new_loss)
                val_mious.append(new_miou)
                '''
                if best_loss is None or new_loss < best_loss:
                    best_loss = new_loss
                    torch.save(model.state_dict(), {args.weights_save})
                    print(f'Loss improved, saving model to {args.weights_save}')
                '''
                if best_miou is None or new_miou > best_miou:
                    best_miou = new_miou
                    torch.save(model.state_dict(), args['weights_save'])
                    print(f'mIoU improved, saving model to {args.weights_save}')
            else:
                torch.save(model.state_dict(), args['weights_save'])
            scheduler.step()
            print()

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].plot(np.arange(1, args.epochs + 1), train_losses, label='Training Loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Epochs')
    if val_losses:
        axes[0].plot(np.arange(1, args.epochs + 1), val_losses, label='Validation Loss')

    axes[1].plot(np.arange(1, args.epochs + 1), all_lr, label='Learning rate')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Learning rate')
    axes[1].set_xlabel('Epochs')

    if val_mious:
        axes[2].plot(np.arange(1, args.epochs + 1), val_mious, label='mIoU')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('mIoU')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()