import torch
import numpy as np
import random
from pytorch_lightning import seed_everything
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
from packages import model_options as MO
from packages import feature_extraction as FE
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda:0')
scaler = StandardScaler()
loss_fn = nn.MSELoss()
SEED = 42
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    seed_everything(SEED)

ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
#-------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

def retrieve_clip_model(model_name):
    import clip;
    model, _ = clip.load(model_name, device='cpu')
    return model.visual

def CLIP_of_generated(generated_frames, model=model_CLIP):
    class TensorDataset(Dataset):
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Z
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

def decode_LDM_latent_feature(model_save_path,recons_img_idx):
    model_name = "fastl2_n_feat_{}.pickle".format(4000)
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()
    pred = model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    z_after_reverse = reverse_reshape_z(pred, mean_y, std_y)
    return z_after_reverse

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# C
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
# CLIP
def decoding_test_1layer(model_save_path,mask_path,layer_name,remain_rate,decoding_function,n, recons_img_idx):
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
        mask = np.load(mask_path + '/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
    else:
        mask = ''

    if decoding_function == 'fastl2':
        pred = (model.predict(x_test))[recons_img_idx:recons_img_idx+1, :]
    else:
        pred = model.predict(x_test).cpu().detach().numpy()

    return pred, mask 

def decoding_target_layers(model_save_path,mask_path,CLIP_mean_path,CLIP_std_path,target_layers,remain_rates,decoding_function,n_feats , recons_img_idx):
    means = np.mean(np.array([np.load(CLIP_mean_path + '/trn_stim_CLIP_mean_{}.npy'.format(i+1)) for i in range(8)]   ) , axis = 0)
    stds = np.mean(np.array([np.load(CLIP_std_path + '/trn_stim_CLIP_std_{}.npy'.format(i+1)) for i in range(8)]   ) , axis = 0)
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
            pred ,mask = decoding_test_1layer(model_save_path,mask_path,layer_name,remain_rate,decoding_function,n ,recons_img_idx)
            pred = pred.squeeze()
            null = np.zeros(38400)
            null_mask = np.zeros(38400)
            for i,index in enumerate(mask):
                null[index] = pred[i]      
                null_mask[index] = 1       

            masks.append(null_mask.reshape(1,-1))

            mean = means[ 38400 * (idx - 1):38400 * idx]
            std = stds[ 38400 * (idx - 1):38400 * idx]
            reverse_pred = ((null*std+mean)*null_mask)[np.newaxis,:]

            dic[layer_name] = reverse_pred
        else:
            pred , _ = decoding_test_1layer(model_save_path,mask_path,layer_name,remain_rate,decoding_function,n ,recons_img_idx)
            pred = pred.squeeze()
            mean = means[38400*12:]
            std = stds[38400 * 12:]
            reverse_pred =  (pred* std + mean)[np.newaxis,:]
            dic[layer_name] = reverse_pred

    return dic , masks
#------------------------------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Structural_feature_extraction')
    parser.add_argument('--target_layers', help='CLIP_layers', default=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12'], type=list)
    parser.add_argument('--val_file_ex',  default='', type=str)  
    parser.add_argument('--SD_config_path',  default='', type=str) 
    parser.add_argument('--SD_ckpt_path',  default='', type=str) 
	
    parser.add_argument('--VAE_feature_decoder_path',  default='', type=str)
    parser.add_argument('--val_VAE_mean_path',  default='', type=str)
    parser.add_argument('--val_VAE_std_path',  default='', type=str)
	
    parser.add_argument('--Text_feature_decoder_path',  default='', type=str)
    parser.add_argument('--val_C_mean_path',  default='', type=str)
    parser.add_argument('--val_C_std_path',  default='', type=str)
    parser.add_argument('--cls_path',  default='', type=str)

    parser.add_argument('--CLIP_feature_decoder_path',  default='', type=str)
    parser.add_argument('--val_CLIP_mean_path',  default='', type=str)
    parser.add_argument('--val_CLIP_std_path',  default='', type=str)
    parser.add_argument('--mask_path',  default='', type=str)

    parser.add_argument('--recons_img_idx,  default='', type=float)
    parser.add_argument('--picture_save_path',  default='', type=str)

    args = parser.parse_args()

    x_test = fetch_ROI_voxel(args.val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
    x_test = scaler.fit_transform(x_test)
	
    #Model_preparation
    #1.CLIP
    model_string = 'ViT-B/32_clip'
    model_option = MO.get_model_options()[model_string]
    image_transforms_CLIP = MO.get_recommended_transforms(model_string)
    model_name = model_option['model_name']
    train_type = model_option['train_type']
    model_CLIP = eval(model_option['call'])
    model_CLIP = model_CLIP.eval()
    model_CLIP = model_CLIP.to(device)
	
    #2.SD
    config = OmegaConf.load(args.SD_config_path)
    model = load_model_from_config(config, args.SD_ckpt_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

    #Z_decoding
    mean_y = (np.load(args.trn_VAE_mean_path)).reshape(1, -1)
    std_y = (np.load(args.trn_VAE_std_path)).reshape(1, -1)
    z = torch.Tensor(decode_LDM_latent_feature(args.VAE_feature_decoder_path, recons_img_idx).astype(np.float32)).to(device)

    #C_decoding
    mean = (np.load(args.val_C_mean_path)).reshape(1, -1)
    std = (np.load(args.val_C_std_path)) # .reshape(1,-1)
    std[:768] = 0
    std = std.reshape(1, -1)
    model_save_path_txt = args.Text_feature_decoder_path
    model_name_txt = "fastl2_n_feat_{}.pickle".format(125)
    f_save_txt = open(model_save_path_txt + model_name_txt, 'rb')
    decode_model = pickle.load(f_save_txt)
    f_save_txt.close()
    cls = np.load(args.cls_path)
    c = torch.tensor(decode_LDM_text_feature_v2(recons_img_idx).astype(np.float32)).to(device)

    #CLIP_decoding
    CLIP_target , feature_masks = decoding_target_layers(args.CLIP_feature_decoder_path, args.mask_path, args.val_CLIP_mean_path,
							 args.val_CLIP_std_path, args.target_layers, 25, 'fastl2',n_feats, args.recons_img_idx)

    if not os.path.exists(args.picture_save_path):
        os.makedirs(args.picture_save_path)
	    
    # Reconstruction
    iterations = 300
    LR = 0.01

    with model.ema_scope():
        uc = model.get_learned_conditioning(1 * [""])

    z = Variable(z, requires_grad=True)
    params = [ z ]

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



if __name__ == "__main__":
    main()
	
