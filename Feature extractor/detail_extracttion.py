import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from torch.autograd import Variable
import PIL
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch.optim import lr_scheduler
from torch.optim import  Adam
from torch import autocast
from tqdm import tqdm
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from packages import model_options as MO
from packages import feature_extraction as FE
from sklearn.preprocessing import StandardScaler
import pickle
import os

seed = 42
seed_everything(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda:0')
scaler = StandardScaler()
loss_fn = nn.MSELoss()
#------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_img(stim,picture_idx):
    image = stim[picture_idx:picture_idx+1, :, :].squeeze()
    image = (255 * image).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image.convert('RGB')
    image = image.resize((512, 512))

    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./stable-diffusion/configs/v1-inference.yaml")
    model = load_model_from_config(config,"./stable-diffusion/checkpoints/sd-v1-4.ckpt")
    return model

#------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_file(stim_path, feature_path):
    stim = np.load(stim_path)
    VAE = []
    for i in tqdm(range(stim.shape[0])):
        img = load_img(stim, i).to(device)
        z = model.get_first_stage_encoding(model.encode_first_stage(img)).cpu().detach().numpy().squeeze()
        VAE.append(z)
    np.save(feature_path, VAE)
    return

def main():
    parser = argparse.ArgumentParser(description='VQVAE feature extraction')
    parser.add_argument('--stable_diffusion_config_path',  default='', type=str)
    parser.add_argument('--stable_diffusion_skpt_path',  default='', type=str)
    parser.add_argument('--stim_saved_path',  default='', type=str)
    parser.add_argument('--feature_saved_path',  default='', type=str)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.stable_diffusion_config_path)
    model = load_model_from_config(config, args.stable_diffusion_skpt_path).to(device)
    sampler = DDIMSampler(model)
    for j in [1]:
        a = save_file(args.stim_saved_path, args.feature_saved_path)
    

if __name__ == "__main__":
    main()






