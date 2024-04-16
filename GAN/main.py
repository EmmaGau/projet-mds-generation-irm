from train import train, create_training_folder
from dataset import ImageFolderDataset, ConcatDataset
from model import Generator, Discriminator
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def main():
    batch_size=64
    name = 'bite-xray' # name of the experiment folder 

    folder =  create_training_folder(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader 
    transformations = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])

    data_folder = 'data/split'
    bitewings_folder = data_folder+'/bitewings/train'
    xray_folder = data_folder+'/xray/train'

    bitewings = ImageFolderDataset(bitewings_folder, transform=transformations)
    xray = ImageFolderDataset(xray_folder, transform=transformations)
    combined_dataset = ConcatDataset([bitewings, xray])

    train_loader = DataLoader(bitewings, batch_size=batch_size, shuffle=True, drop_last=True)

    # model 
    # Create some generator and discriminator
    nz = 100
    G = Generator().to(device)
    D = Discriminator().to(device)

    # train, choose arguments 
    train(D,
        G,
        train_loader=train_loader,
        folder=folder,
        device=device,
        batch_size=batch_size,
        nz=nz,
        num_epochs=1000,
        gpw=0.1,
        lr=0.0001,
        log_every=0,
        save_every=100)
    
if __name__ == '__main__':
    main()