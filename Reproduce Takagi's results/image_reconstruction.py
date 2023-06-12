import torch
import numpy as np
import random
SEED = 816
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

from omegaconf import OmegaConf
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
from torch import nn
loss_fn = nn.MSELoss()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda:0')

#-------------------------------------------------------------------------------------------------------------------------------------------------
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'
stim = np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_multi_trial_data.npy')

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
x_test = scaler.fit_transform(x_test)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#对z的解码
def reshape(l):
#step1:(8859,4,64,64)--→(8859,4,4096)
    c = []
    for k in range(l.shape[0]):
        b = []
        for i in range(l.shape[1]):
            a = np.concatenate([l[k, i, j, :] for j in range(l.shape[2])], axis=0)
            b.append(a)
        c.append(b)
    c = np.array(c)
#step2:(8859,4,4096)--→(8859,4*4096)
    d = []
    for i in range(c.shape[0]):
        a = np.concatenate([c[i, j, :] for j in range(c.shape[1])], axis=0)
        d.append(a)
    d = np.array(d)
    return d

def reverse_reshape_z(d, mean, std):
    d = d * std + mean
    b = []
    for i in range(d.shape[0]):
        a = np.concatenate([(d[i])[np.newaxis, 4096 * j:4096 * (j + 1)] for j in range(4)], axis=0)
        b.append(a)
    b_after_reverse = np.array(b)
    print(b_after_reverse.shape)

    e = []
    for i in range(b_after_reverse.shape[0]):
        a = b_after_reverse[i]
        f = []
        for j in range(a.shape[0]):
            s = np.concatenate([(a[j])[np.newaxis, 64 * k:64 * (k + 1)] for k in range(64)], axis=0)
            f.append(s)
        f = np.array(f)
        e.append(f)
    b_reverse = np.array(e)
    return b_reverse
l = np.load('./图像重建/数据集/Stable-diffusion隐空间特征/trn_stim_lattent_z.npy')
y = reshape(l)
mean_y = np.mean(y,axis=0).reshape(1, -1)
std_y = np.std(y,axis=0).reshape(1, -1)

def decode_LDM_latent_feature(recons_img_idx):
    #x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
    #x_test = scaler.fit_transform(x_test)
    model_save_path = './图像重建/数据集/Stable-diffusion隐空间特征/'
    model_name = "fastl2_n_feat_{}.pickle".format(4000)
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()
    pred = model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    z_after_reverse = reverse_reshape_z(pred, mean_y, std_y)
    return z_after_reverse

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#对c的解码

mean = (np.load('./图像重建/数据集/Stable-diffusion文本特征/LDM_15_average_mean.npy')).reshape(1, -1)
std = (np.load('./图像重建/数据集/Stable-diffusion文本特征/LDM_15_average_std.npy')) # .reshape(1,-1)
std[:768] = 0
std = std.reshape(1, -1)
model_save_path_txt = './图像重建/数据集/解码到stable-diffusion的文本空间/降为15维/同时拟合/提取的average特征/'
model_name_txt = "fastl2_n_feat_{}.pickle".format(250)
f_save_txt = open(model_save_path_txt + model_name_txt, 'rb')
decode_model = pickle.load(f_save_txt)
f_save_txt.close()
cls = np.load('./图像重建/数据集/Stable-diffusion文本特征/LDM_10_cls.npy')

def _reshape(l):  #(8859,15,768)>>(8859,15*768)
	b = []
	for i in range(l.shape[0]):
		a = np.concatenate([l[i,j,:] for j in range(l.shape[1])], axis=0)
		b.append(a)
	b = np.array(b)
	return b

def reverse_reshape(l , mean, std): #(8859,15*768)>>(8859,15,768);revers_z_score
    b = []
    l = l * std + mean
    for i in range(l.shape[0]):
        a = np.concatenate([(l[i])[np.newaxis, 768 * j:768 * (j + 1)] for j in range(15)], axis=0)
        b.append(a)
    b_after_reverse = np.array(b)
    return b_after_reverse

def decode_LDM_text_feature_v2(recons_img_idx):
    pred = decode_model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    z = np.zeros((1,11520))
    z[:,:768] = cls
    z[:,768:] = pred
    z_after_reverse = reverse_reshape(z, mean, std)
    return z_after_reverse

#-------------------------------------------------------------------------------------------------------------------------------------------------


def recons_withz_c(recons_img_idx):
    picture_save_path = './stable-diffusion/实验结果/复现Nishimoto对z和c解码'
    if not os.path.exists(picture_save_path):
        os.makedirs(picture_save_path)

    z = torch.Tensor(decode_LDM_latent_feature(recons_img_idx).astype(np.float32)).to(device)
    c = torch.tensor(decode_LDM_text_feature_v2(recons_img_idx).astype(np.float32)).to(device)

    z = model.decode_first_stage(z)
    z = torch.clamp((z + 1.0) / 2.0, min=0.0, max=1.0)
    z = model.get_first_stage_encoding(model.encode_first_stage(z)).cpu().detach().numpy()
    z = torch.Tensor(z.astype(np.float32)).to(device)

    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
    z_enc = sampler.stochastic_encode(z, torch.tensor([37] * 1).to(device))
    with model.ema_scope():
        uc = model.get_learned_conditioning(1 * [""])

    samples = sampler.decode(z_enc, c, 35, unconditional_guidance_scale=5.0, unconditional_conditioning=uc, )
    x_samples_ddim = model.decode_first_stage(samples)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().detach().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(picture_save_path+'/{}_rec_z_c.png'.format(recons_img_idx) )

    image = stim[recons_img_idx:recons_img_idx + 1, :, :].squeeze()
    image = (255 * image).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image.convert('RGB')
    image = image.resize((512, 512))
    image.save(picture_save_path + '/{}_real.png'.format(recons_img_idx))
    return

for i in range(982):#7,10,44,104,121,579,265,325,517
    a = recons_withz_c(i)
    print("第{}张图像重建完毕".format(i))









