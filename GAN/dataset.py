import os
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms
from PIL import Image


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Chemin vers le dossier contenant les images.
            transform (callable, optional): Transformation optionnelle à appliquer sur les échantillons.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        # chechecker la taille de l'image
        if image.size != (256, 256):
            image = image.resize((256, 256))
        if self.transform:
            image = self.transform(image)
        return image



if __name__== "__main__":
    # Définir les transformations que vous souhaitez appliquer sur les images
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ])

    # Créer une instance de votre Dataset
    bitewings_folder = 'SeGan/data/Bitewings_Resized_256'
    xray_folder = 'SeGan/data/x-ray_Resized_256'
    # fusionner les 2 
    bitewings = ImageFolderDataset(bitewings_folder, transform=transformations)
    xray = ImageFolderDataset(xray_folder, transform=transformations)
    combined_dataset = ConcatDataset([bitewings, xray])
    
    

    