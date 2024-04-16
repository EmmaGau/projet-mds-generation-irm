
# from pytorch_fid import fid_score
import torch
import numpy as np
from PIL import Image
import torchvision
from model import Generator
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import os 
import torch
from PIL import Image
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
from model import Generator
import torchvision.models as models
from pytorch_fid import fid_score
import lpips

def pilimg_to_tensor(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Change according to your settings
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image)

class Evaluation:
    def __init__(self, checkpoint_path, save_dir, path_real_images):
        self.checkpoint_path = checkpoint_path
        # create save dir
        self.save_dir = os.path.join(save_dir, self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1])
        os.makedirs(self.save_dir, exist_ok=True)
        # load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = self.load_model()
        self.path_real_images = path_real_images
        self.nb_images = len(os.listdir(path_real_images))
            
    def load_model(self):
        G = Generator()
        G.load_state_dict(torch.load(self.checkpoint_path,map_location=self.device))
        G.to(self.device)
        G.eval()
        return G
    
    def psnr(self,uref,ut,M=1):
        rmse = np.sqrt(np.mean((np.array(uref).cpu()-np.array(ut).cpu())**2))
        return 20*np.log10(M/rmse)

    def ssim(self,uref,ut):
        uref = uref.numpy().cpu()
        ut = ut.numpy().cpu()
        return compare_ssim(uref,ut,multichannel=True,channel_axis=0,data_range= 2) # as images are normalized to [-1,1]

    def compute_metrics(self, generated_path, real_path):
        psnr_values = []
        ssim_values = []
        real_images_list = os.listdir(real_path)
        for i in range(self.nb_images):
            img_path_gen = os.path.join(generated_path, f'{str(i).zfill(5)}.png')
            img_path_real = os.path.join(real_path, real_images_list[i])
            x1 = pilimg_to_tensor(Image.open(img_path_gen)).unsqueeze(0).to(self.device)
            x2 = pilimg_to_tensor(Image.open(img_path_real)).unsqueeze(0).to(self.device)
            print(x1.shape, x2.shape)
            print(type(x1), type(x2))
            psnr_values.append(self.psnr(x1, x2).item())
            ssim_values.append(self.ssim(x1.squeeze(), x2.squeeze()))
        return np.mean(psnr_values), np.mean(ssim_values)

    
    def generate_images(self, nb_images=1000, nz=100):
        z = torch.randn(nb_images, nz, 1, 1, device=self.device)
        with torch.no_grad():
            images = self.generator(z)
        for i, image in enumerate(images):
            utils.save_image(image, os.path.join(self.save_dir, f'{str(i).zfill(5)}.png'))

    def calculate_fid(self, path_real_images):
        fid_value = fid_score.calculate_fid_given_paths(
            [self.save_dir, path_real_images],
            batch_size=50,
            device=str(self.device),
            dims=2048,
            num_workers=4
        )
        print("FID:", fid_value)
        return fid_value

    def inception_score(self, nb_images, splits=10):
        inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception_model.to(self.device)
        inception_model.eval()
        
        def get_pred(x):
            if x.size(1) != 3:
                x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            x = transforms.functional.resize(x, (299, 299))
            return inception_model(x)

        imgs = [pilimg_to_tensor(Image.open(os.path.join(self.save_dir, f'{str(i).zfill(5)}.png'))) for i in range(nb_images)]
        imgs = torch.stack(imgs).to(self.device)
        preds = torch.softmax(get_pred(imgs), dim=1)
        
        split_scores = []
        for i in range(splits):
            part = preds[i * (nb_images // splits): (i + 1) * (nb_images // splits), :]
            kl_div = part * (torch.log(part) - torch.log(torch.mean(part, dim=0)))
            split_scores.append(torch.exp(torch.mean(torch.sum(kl_div, dim=1))))
        
        return torch.mean(torch.tensor(split_scores)), torch.std(torch.tensor(split_scores))

    def evaluate_model(self):
        nb_generated_img = 1000
        # self.generate_images(nb_generated_img)
        fid_value = self.calculate_fid(self.path_real_images)
        inception_mean, inception_std = self.inception_score(nb_generated_img)
        psnr_avg, ssim_avg = self.compute_metrics(self.save_dir, self.path_real_images)
        # load metrics in a csv
        with open(os.path.join(self.save_dir, 'metrics.csv'), 'w') as f:
            f.write('FID,Inception_mean,Inception_std, PSNR, SSIM\n')
            f.write(f'{fid_value},{inception_mean},{inception_std}, {psnr_avg}, {ssim_avg}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Generative Model")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--save_dir", required=True, help="Directory to save generated images and results.")
    parser.add_argument("--path_real_images", required=True, help="Path to directory with real images for FID calculation.")
    args = parser.parse_args()
    
    evaluator = Evaluation(args.checkpoint_path, args.save_dir, args.path_real_images)
    evaluator.evaluate_model()
