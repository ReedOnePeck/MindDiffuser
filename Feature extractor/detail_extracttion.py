import torch
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
loss_fn = nn.MSELoss()

from packages import model_options as MO
from packages import feature_extraction as FE

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda:0')
seed = 42
seed_everything(seed)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
#trn_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_'
#val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

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
#一、模型准备
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


model = get_model()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
#------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_file(j):
    stim = np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj0{}/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_multi_trial_data.npy'.format(j))
    VAE = []
    for i in tqdm(range(982)):
        img = load_img(stim, i).to(device)
        z = model.get_first_stage_encoding(model.encode_first_stage(img)).cpu().detach().numpy().squeeze()
        VAE.append(z)
    np.save('/val_stim_lattent_z.npy'.format(j),VAE)
    print("被试{}数据已处理完毕".format(j))
    return

for j in [1]:
    a = save_file(j)
print('')
#------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

z_enc = sampler.stochastic_encode(z, torch.tensor([37]*1).to(device))
with model.ema_scope():
    uc = model.get_learned_conditioning(1 * [""])

prompt = "a green airplane"

c = model.get_learned_conditioning([prompt])
samples = sampler.decode(z_enc, c, 35, unconditional_guidance_scale=5.0,unconditional_conditioning=uc,)

picture_save_path = './stable-diffusion/实验结果/15_768以及反向传播/picture_10'

x_samples_ddim = model.decode_first_stage(samples)
x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
grid = make_grid(grid, nrow=1)
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().detach().numpy()
Image.fromarray(grid.astype(np.uint8)).save('%s/rec_iter.png' % (picture_save_path))

print('')

"""





