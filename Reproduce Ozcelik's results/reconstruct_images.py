from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
#import fastl2lir
import numpy as np
import sys
sys.path.append('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan')
import ic_gan.inference.utils as inference_utils
from torch import nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda:0')
from torch.cuda.amp import autocast as autocast
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#加载模型
def replace_to_inplace_relu(model):  # saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
  for child_name, child in model.named_children():
    if isinstance(child, nn.ReLU):
      setattr(model, child_name, nn.ReLU(inplace=False))
    else:
      replace_to_inplace_relu(child)
  return

def load_icgan(experiment_name, root_ = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan/pretrained_models'):
  root = os.path.join(root_, experiment_name)
  config = torch.load("%s/%s.pth" %(root, "state_dict_best0"))['config']

  config["weights_root"] = root_
  config["model_backbone"] = 'biggan'
  config["experiment_name"] = experiment_name
  G, config = inference_utils.load_model_inference(config)
  G.cuda()
  G.eval()
  return G

experiment_name = 'icgan_biggan_imagenet_res256'
ic_model = load_icgan(experiment_name)
replace_to_inplace_relu(ic_model)
eps = 1e-8
print('Model is loaded')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#加载数据
#x
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
#trn_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_'
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'

def z_score(data):
    scaler = StandardScaler()
    data_=scaler.fit_transform(data)
    mean=np.mean(data,axis=0)
    s=np.std(data,axis=0)
    return data_,mean,s

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i] ) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

x = fetch_ROI_voxel(val_file_ex, ROIs)  # (8859,11694)
x = scaler.fit_transform(x)


mean = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/mean.npy')#.reshape(1,-1)
std = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/std.npy')#.reshape(1,-1)

model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/'

model_name = "fastl2_n_feat_{}.pickle".format(200)
f_save = open(model_save_path + model_name, 'rb')
model = pickle.load(f_save)
f_save.close()

def recons_ic_gan(idx):

    pred = model.predict(x)[idx:idx + 1, :]
    I = torch.tensor((pred*std+mean).astype(np.float32)).to(device)
    n = torch.tensor(np.random.randn(1, 119).astype(np.float32)).to(device)
    with autocast():
        out = ic_model(n, None, I)
    out = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

    grid = rearrange(out, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().detach().numpy()
    Image.fromarray(grid.astype(np.uint8)).save('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/重建结果/测试集重建/test{}.png'.format(idx))
    return

for i in tqdm(range(982)):
    a = recons_ic_gan(idx = i)






