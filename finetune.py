from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    diffusion_defaults
)
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import os
import cv2
import albumentations as A
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import tqdm
import torch
from guided_diffusion.unet import FinetuneUNetModel, UNetModel
import argparse
from dataloader import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torchmetrics


# Les deux fonctions ci dessous sont recopiées de guided diffusion
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
        # Seule différence ici, dans guided diffusion il y avait 3 si learn sigma = False else 6
        # mais on veut avoir X calques de sortie pour segmenter
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


# Idem, recopié, j'aurais pu faire un import mais bon
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
    return parser


def main():
    # Rajout des paramètres pour segmenter
    parser = create_argparser()
    parser.add_argument('model_path')
    parser.add_argument('img_dir')
    parser.add_argument('mask_dir')
    parser.add_argument('--imgval_dir', default=None)
    parser.add_argument('--maskval_dir', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--weights_save', default='unet.pth')
    parser.add_argument('--class_weighting', action='store_true')
    parser.add_argument('--unfreeze_decoder', type=float, default=1)
    args = parser.parse_args()

    assert 0. <= args.unfreeze_decoder <= 1.
    # # Print pour vérifier
    # for k, v in args.__dict__.items():
    #     print(k, v)

    dist_util.setup_dist()
    # logger.configure()
    model = create_model(
        args.image_size,
        args.num_channels,
        args.num_res_blocks,
        args.channel_mult,
        False,  # learn sigma, mais on ne s'en sert pas
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

    state_dict = torch.load(args.model_path)

    # Comme le modèle qu'on veut finetuner a un nombre différent de couches de sortie, on ne peut pas load les poids car sinon
    # shape mismatch, on delete donc la couche de sortie
    del state_dict['out.2.weight']
    del state_dict['out.2.bias']
    model.load_state_dict(
        state_dict, strict=False
    )

    device = 'cuda'
    model = model.to(device)

    # Vérification du loading du modèle
    model_layer_names = [name for name, _ in model.named_parameters()]
    state_dict_layer_names = list(state_dict.keys())
    for name, param in model.named_parameters():
        if name in state_dict:
            pretrained_weights = state_dict[name]
            if torch.allclose(param.data, pretrained_weights):
                pass
            else:
                print(f"Les poids de la couche {name} n'ont pas été chargés correctement.")
        else:
            print(f"La couche {name} n'a pas été trouvée dans le modèle pré-entraîné.")

    params_train = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 12}

    # Augmentation
    transform = A.Compose([
        A.RandomCrop(width=args.image_size, height=args.image_size),
        A.Affine(translate_percent=(0.05, 0.05), rotate=(-180, 180), shear=(-5, 5), scale=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

    dataset = Dataset(args.img_dir, args.mask_dir, transform, args.image_size)
    training_generator = torch.utils.data.DataLoader(dataset, **params_train)

    # print the first element of the training set
    img = next(iter(training_generator))
    print(img[0].size(), img[1].size())

    validation_generator = None
    if args.imgval_dir is not None and args.maskval_dir is not None:
        valdata = Dataset(args.imgval_dir, args.maskval_dir, None, args.image_size)
        params_val = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 12}

        validation_generator = torch.utils.data.DataLoader(valdata, **params_val)

    train_losses = []
    val_losses = []
    val_mious = []
    all_lr = []
    best_loss = None
    best_miou = None

    # Class weighting
    print('Calculating class weights ...')
    histo = {i: 0 for i in range(args.out_channels)}
    for mask_path in os.listdir(args.mask_dir):
        m = cv2.imread(os.path.join(args.mask_dir, mask_path), -1)
        unique_values, unique_counts = np.unique(np.array(m).flatten(), return_counts=True)
        for val, c in zip(unique_values, unique_counts):
            histo[val] += c

    weights = []
    total = np.array(list(histo.values())).sum()
    for i in range(args.out_channels):
        weights.append(1 - histo[i] / total)
    weights = torch.FloatTensor(weights).to(device)
    print('Done ! ', weights)

    criterion = torch.nn.CrossEntropyLoss(weight=weights if args.class_weighting else None)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 30, T_mult = 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    unfreeze_decoder = int(args.epochs * args.unfreeze_decoder)  # nombre d'epochs à partir duquel on va defreeze le decoder
    if validation_generator is not None:
        print(f'{len(training_generator)} samples in training, {len(validation_generator)} samples in validation')
    else:
        print(f'{len(training_generator)} samples in training, no validation dataset')
    print(f'Unfreeze decoder after {unfreeze_decoder} epochs')


    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=args.out_channels).to(device)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = []
        start = time.time()

        all_lr.append(scheduler.state_dict()['_last_lr'])
        for img, mask in tqdm.tqdm(training_generator, total=len(training_generator)):
            img, mask = img.to(device), mask.to(device)
            # print(img.size(), img.min(), img.max(), img.type(), mask.size(), mask.min(), mask.max(), mask.type())
            # zero the parameter gradients
            optimizer.zero_grad()

            # On donne l'image d'origine avec un timestep fixe à 1 (pour simuler une image quasiment pas bruitée pour
            # rester cohérent avec le pretrain)
            outputs = model(img, timesteps=torch.tensor([1.], dtype=torch.float32, device=device))

            loss = criterion(outputs, mask.long())

            # Vérification de la GT en nuances de gris (+ prédictions éventuelles)
            if args.show:
                print(f'Outputs shape {outputs.size()} Target shape {mask.size()}')
                logsoftmax = torch.nn.functional.log_softmax(outputs, dim=1)
                viz = torch.argmax(logsoftmax, dim=1)

                img_show = img.detach().cpu().numpy()
                mask_show = mask.detach().cpu().numpy()
                # print(img_show.shape, mask_show.shape)
                fig, axes = plt.subplots(nrows=img.size()[0], ncols=3)

                for index, i_ in enumerate(img_show):
                    try:

                        axes[index, 0].set_title('Input')
                        axes[index, 1].set_title('GT')
                        axes[index, 2].set_title('Pred')

                        i_ = i_.transpose(1, 2, 0)
                        i_ = ((i_ + 1.) * 127.5).astype(np.uint8)
                        m_ = mask_show[index]
                        v_ = viz.detach().cpu().numpy()[index]
                        multiply = int(256 / args.out_channels)
                        m_ = m_ * multiply
                        v_ = v_ * multiply
                        axes[index][0].imshow(i_, cmap='gray')
                        axes[index][1].imshow(m_, cmap='gray', vmin=0, vmax=255)
                        axes[index][2].imshow(v_, cmap='gray', vmin=0, vmax=255)
                    except Exception as e:
                        print(e)
                        continue
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.show()

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
                    outputs = model(img, timesteps=torch.tensor([1.], dtype=torch.float32, device=device))
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
                torch.save(model.state_dict(), args.weights_save)
                print(f'mIoU improved, saving model to {args.weights_save}')
        else:
            torch.save(model.state_dict(), args.weights_save)
        scheduler.step()
        print()

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].plot(np.arange(1, args.epochs + 1), train_losses, label='Training Loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Epochs')
    if val_losses:
        axes[0].plot(np.arange(1, args.epochs + 1), val_losses, label='Validation Loss')
    if args.unfreeze_decoder != 1.:
        axes[0].axvline(unfreeze_decoder, color='red', linestyle='--')

    axes[1].plot(np.arange(1, args.epochs + 1), all_lr, label='Learning rate')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Learning rate')
    axes[1].set_xlabel('Epochs')
    if args.unfreeze_decoder != 1.:
        axes[1].axvline(unfreeze_decoder, color='red', linestyle='--')

    if val_mious:
        axes[2].plot(np.arange(1, args.epochs + 1), val_mious, label='mIoU')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('mIoU')
    if args.unfreeze_decoder != 1.:
        axes[2].axvline(unfreeze_decoder, color='red', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
