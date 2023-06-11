import torch
import numpy as np
import random

SEED = 42
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
from pytorch_lightning import seed_everything
seed = 42
seed_everything(seed)

from omegaconf import OmegaConf
from torch.autograd import Variable
import PIL
from einops import rearrange
from torchvision.utils import make_grid

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


#-------------------------------------------------------------------------------------------------------------------------------------------------
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'
#stim = np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_multi_trial_data.npy')

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
x_test = scaler.fit_transform(x_test)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#一、模型准备
#（1）CLIP
model_string = 'ViT-B/32_clip'
model_option = MO.get_model_options()[model_string]
image_transforms_CLIP = MO.get_recommended_transforms(model_string)
model_name = model_option['model_name']
train_type = model_option['train_type']


def retrieve_clip_model(model_name):
    import clip;
    model, _ = clip.load(model_name, device='cpu')
    return model.visual


model_CLIP = eval(model_option['call'])
model_CLIP = model_CLIP.eval()
model_CLIP = model_CLIP.to(device)


def CLIP_of_generated(generated_frames, model=model_CLIP):
    class TensorDataset(Dataset):
        # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
        def __init__(self, data_tensor, image_transforms=None):
            self.data_tensor = data_tensor
            self.transforms = image_transforms

        def __getitem__(self, index):
            img = self.data_tensor[index]
            if self.transforms:
                img = self.transforms(img)
            return img#.to(torch.float16)

        def __len__(self):
            return self.data_tensor.size(0)

    transform_for_CLIP = torchvision.transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    stimulus_loader = DataLoader(TensorDataset(generated_frames, transform_for_CLIP), batch_size=32)

    target_layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8','Linear-10','Linear-12', 'VisionTransformer-1']

    feature_maps = FE.get_all_feature_maps(model, stimulus_loader, layers_to_retain=target_layers,
                                           remove_duplicates=False, numpy=False)
    for feature_map in feature_maps.keys():
        feature_maps[feature_map] = feature_maps[feature_map].to(device)

    return feature_maps


def MSE_CLIP(target, generated, weight):
    MSE = 0
    for feature_map in generated.keys():
        if feature_map != 'VisionTransformer-1':
            MSEi = (1 - weight) * loss_fn(target[feature_map], generated[feature_map])
            MSE = MSE + MSEi
        else:
            MSEi = 1. - torch.cosine_similarity(target[feature_map], generated[feature_map])
            MSE = MSE + 0.1 * weight * MSEi
    return MSE

def MSE_CLIP_mask(masks, target, generated, weight):
    MSE = []
    VIT = []
    norms = []
    for layer in target.keys():
        if layer != 'VisionTransformer-1':
            norms.append(np.linalg.norm(target[layer]))

    norms = np.array(norms)
    b = 1./norms
    weights = b/b.sum()

    a1 = np.array([1.5,1.2,1,1,1,1])
    weights = weights*a1

    for idx , feature_map in enumerate(generated.keys()):
        if feature_map!= 'VisionTransformer-1':
            MSEi = (1-weight)*loss_fn(torch.tensor(target[feature_map]).to(device) , generated[feature_map]*((torch.tensor(masks[idx])).to(device)))
            MSE.append(MSEi)
        else:
            MSEi = 1. - torch.cosine_similarity(torch.tensor(target[feature_map]).to(device) , generated[feature_map])
            VIT.append(0.1*weight*MSEi)
    mse = VIT[0]
    for i in range(weights.shape[0]):
        mse = mse+MSE[i]*(torch.tensor(weights[i]).to(device))
    return mse
#(2)Stable diffusion
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
    config = OmegaConf.load("/nfs/diskstation/DataStation/ChangdeDu/LYZ/stable-diffusion/configs/v1-inference.yaml")
    model = load_model_from_config(config,"/nfs/diskstation/DataStation/ChangdeDu/LYZ/stable-diffusion/checkpoints/sd-v1-4.ckpt")
    return model

model = get_model()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
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
l = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/trn_stim_lattent_z.npy')
y = reshape(l)
#np.save('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/mean_y.npy',np.mean(y,axis=0))
#np.save('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/std_y.npy',np.std(y,axis=0))
mean_y = (np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/mean_y.npy')).reshape(1, -1)
std_y = (np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/std_y.npy')).reshape(1, -1)


def decode_LDM_latent_feature(recons_img_idx):
    #x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
    #x_test = scaler.fit_transform(x_test)
    model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/'
    model_name = "fastl2_n_feat_{}.pickle".format(4000)
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()
    pred = model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    z_after_reverse = reverse_reshape_z(pred, mean_y, std_y)
    return z_after_reverse

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#对c的解码

mean = (np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion文本特征/LDM_15_best_mean.npy')).reshape(1, -1)
std = (np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion文本特征/LDM_15_best_std.npy')) # .reshape(1,-1)
std[:768] = 0
std = std.reshape(1, -1)
model_save_path_txt = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/解码到stable-diffusion的文本空间/降为15维/同时拟合/提取的best特征/'
model_name_txt = "fastl2_n_feat_{}.pickle".format(125)
f_save_txt = open(model_save_path_txt + model_name_txt, 'rb')
decode_model = pickle.load(f_save_txt)
f_save_txt.close()
cls = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion文本特征/LDM_10_cls.npy')

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
#对CLIP特征的解码
def decoding_test_1layer(layer_name,remain_rate,decoding_function,n, recons_img_idx):
    model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/解码特征选取/{}/解码模型权重保存/{}_remain_{}/'.format(layer_name,decoding_function,remain_rate)

    if decoding_function == 'fastl2':
        model_name = "fastl2_n_feat_{}.pickle".format(n)
    else:
        model_name = "himalaya.pickle"
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()

    x_test = fetch_ROI_voxel(val_file_ex, ROIs)
    x_test = scaler.fit_transform(x_test)

    if layer_name != 'VisionTransformer-1':
        mask = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/解码特征选取/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
    else:
        mask = ''

    if decoding_function == 'fastl2':
        pred = (model.predict(x_test))[recons_img_idx:recons_img_idx+1, :]
    else:
        pred = model.predict(x_test).cpu().detach().numpy()

    return pred, mask #这是zscore后的，别忘了乘以标准差，加均值

def decoding_target_layers(target_layers,remain_rates,decoding_function,n_feats , recons_img_idx):
    """
    :param target_layers: 对不同的图像重建时，使用的特征层不同，这里以列表的形式传入
    :param remain_rates: 对不同的特征进行解码的时候，保留的特征比例不同，每一层的特征比例以列表的形式传入. 对于CLIP的最后一层，恒为100
    :param decoding_function: 解码器，默认为fastl2lir
    :param n_feats:在对每一层CLIP视觉特征进行解码的时候，用来拟合的体素数量不同，数量也以列表的形式传入
    :return:返回一个    反z_scroe变换    后的解码特征，格式是字典
    """
    #首先提取验证集的x_val[500:,:],对其进行z_score

    means = np.mean(np.array([np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_CLIP_mean_{}.npy'.format(i+1)) for i in range(8)]   ) , axis = 0)
    stds = np.mean(np.array([np.load('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_CLIP_std_{}.npy'.format(i+1)) for i in range(8)]   ) , axis = 0)
    #逐层解码，这里对每一层要设置的参数有保留的特征的百分比、筛选的体素的数量（传入一个与layers等长的列表来表示），默认全部使用fastl2lir进行解码
    dic = dict()
    masks = []
    for i in range(len(target_layers)):
        layer_name = target_layers[i]
        remain_rate = remain_rates[i]
        n = n_feats[i]
        if layer_name != 'VisionTransformer-1':
            if layer_name[-2] == '-':
                idx = int(int(layer_name[-1]) / 2)
            else:
                idx = int(int(layer_name[-2:]) / 2)
            pred ,mask = decoding_test_1layer(layer_name,remain_rate,decoding_function,n ,recons_img_idx)
            pred = pred.squeeze()
            null = np.zeros(38400)
            null_mask = np.zeros(38400)
            for i,index in enumerate(mask):
                null[index] = pred[i]       #把解码出来的特征按照mask里存储的索引填入38400维的0向量中恢复其维度
                null_mask[index] = 1        #由于在反z_score变换中,会加上均值，因此变换后再乘以一个null_mask来掩盖掉

            masks.append(null_mask.reshape(1,-1))

            mean = means[ 38400 * (idx - 1):38400 * idx]
            std = stds[ 38400 * (idx - 1):38400 * idx]
            reverse_pred = ((null*std+mean)*null_mask)[np.newaxis,:]

            dic[layer_name] = reverse_pred
        else:
            pred , _ = decoding_test_1layer(layer_name,remain_rate,decoding_function,n ,recons_img_idx)
            pred = pred.squeeze()
            mean = means[38400*12:]
            std = stds[38400 * 12:]
            reverse_pred =  (pred* std + mean)[np.newaxis,:]
            dic[layer_name] = reverse_pred

    return dic , masks
#-------------------------------------------------------------------------------------------------------------------------------------------
def decode_from_fMRI(target_layers,remain_rates,decoding_function,n_feats_img,loss_CLIP_weight , recons_img_idx):
    """
    :param target_layers: 对不同的图像重建时，使用的特征层不同，这里以列表的形式传入
    :param remain_rates: 对不同的特征进行解码的时候，保留的特征比例不同，每一层的特征比例以列表的形式传入. 对于CLIP的最后一层，恒为100
    :param decoding_function: 解码器，默认为fastl2lir
    :param n_feats_img: 在对每一层CLIP视觉特征进行解码的时候，用来拟合的体素数量不同，数量也以列表的形式传入
    :param n_feat_text: 对文本信息进行解码的时候，也会采用不同的体素数量
    :param loss_CLIP_weight: 重建时的监督信号中， CLIP最后一层特征的占比（0~1）
    :param recons_img_idx: 待重建的图像的标号（在测试集中的标号）
    :param exp_number:第几次实验，每次实验时都要把相应的参数保存下来方便对比
    """

    z = torch.Tensor(decode_LDM_latent_feature(recons_img_idx).astype(np.float32)).to(device)
    c = torch.randn((1, 15, 768)).to(device)
    #c = torch.tensor(decode_LDM_text_feature_v2(recons_img_idx).astype(np.float32)).to(device)
    CLIP_target , feature_masks = decoding_target_layers(target_layers,remain_rates,decoding_function,n_feats_img , recons_img_idx)

    picture_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/stable-diffusion/实验结果/消融实验/对c消融/without_c/picture_{}'.format(recons_img_idx)
    if not os.path.exists(picture_save_path):
        os.makedirs(picture_save_path)
    # Reconstruction
    iterations = 300
    LR = 0.01

    with model.ema_scope():
        uc = model.get_learned_conditioning(1 * [""])

    z = Variable(z, requires_grad=True)
    c = Variable(c, requires_grad=True)
    params = [  c,z ]

    optimizer = Adam(params, lr=LR)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for iter in range(iterations):
        scheduler.step()

        z_enc = sampler.stochastic_encode(z, torch.tensor([37] * 1).to(device))
        samples = sampler.decode(z_enc, c, 35, unconditional_guidance_scale=5.0, unconditional_conditioning=uc, )
        x_samples_ddim = model.decode_first_stage(samples)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        optimizer.zero_grad()
        CLIP_generated = CLIP_of_generated(generated_frames=x_samples_ddim)
        loss = MSE_CLIP_mask(masks=feature_masks, target=CLIP_target, generated=CLIP_generated,
                             weight=loss_CLIP_weight)

        loss.backward(retain_graph=False)
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        if iter % 10 == 0:
            # visualize reconstructions
            grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
            grid = make_grid(grid, nrow=1)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().detach().numpy()
            Image.fromarray(grid.astype(np.uint8)).save('%s/rec_iter_%03d.png' % (picture_save_path, iter))
            print('===============================================================> Iter: {:03d} LR: {:.4f} Train loss: {:.4f}'.format(iter, lr, loss.item()))

        del loss, CLIP_generated, x_samples_ddim, samples
        torch.cuda.empty_cache()

    return

#从fMRI解码并重建图像刺激
target_layers = [ 'Linear-2','Linear-4','Linear-6','Linear-8','Linear-10','Linear-12', 'VisionTransformer-1']
remain_rates = [25,25,25,25,25,25,100]
decoding_function = 'fastl2'
n_feats = [350,350,350,250,250,250,450]
 
for idx in [44,104,121]:#range(241,300): #511,517,      576,584,586
    c = decode_from_fMRI(target_layers=target_layers, remain_rates=remain_rates, decoding_function=decoding_function,
                         n_feats_img=n_feats, loss_CLIP_weight=0, recons_img_idx=idx)
print('')










