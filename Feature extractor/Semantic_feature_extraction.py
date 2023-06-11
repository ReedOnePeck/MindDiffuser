from pycocotools.coco import COCO
import os
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda:0')
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import clip
from packages import model_options as MO
from packages import feature_extraction_detach as FE
import sys
sys.path.append('/data/home/cddu/pythonProject/code/taming-transformers')
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#LDM Model
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
    model = load_model_from_config(config, "./stable-diffusion/checkpoints/sd-v1-4.ckpt")
    return model

LDM_model = get_model().to(device)
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#prompt = "a man is skate boarding down a path and a dog is running by his side"
#c = LDM_model.get_learned_conditioning([prompt])
#------------------------------------------------------------------------------------------------------------------------------------------------------------
#CLIP  Model
#https://github.com/openai/CLIP
"""

clip_model,_ = clip.load("ViT-B/32", device=device)    #用来提取文本特征


#用来提取图像特征
model_string = 'ViT-B/32_clip'
model_option = MO.get_model_options()[model_string]
image_transforms = MO.get_recommended_transforms(model_string)
model_name = model_option['model_name']
train_type = model_option['train_type']

def retrieve_clip_model(model_name):
    import clip;
    model, _ = clip.load(model_name, device='cpu')
    return model.visual

model = eval(model_option['call'])
model = model.eval()
model = model.to(device)

def CLIP_extraction(data_tensor):
    transform_for_CLIP = torchvision.transforms.Compose([
        #transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    ])

    class TensorDataset(Dataset):
        def __init__(self, data_tensor, image_transforms=None):
            self.data_tensor = data_tensor
            self.transforms = image_transforms

        def __getitem__(self, index):
            img = self.data_tensor[index]
            if self.transforms:
                img = self.transforms(img)
            return img

        def __len__(self):
            return self.data_tensor.size(0)


    stimulus_loader = DataLoader(TensorDataset(data_tensor, transform_for_CLIP), batch_size=1)

    target_layers = ['VisionTransformer-1']
    feature_maps = FE.get_all_feature_maps(model, stimulus_loader, layers_to_retain=target_layers,
                                           remove_duplicates=False, numpy=False)
    return feature_maps['VisionTransformer-1']
"""
#------------------------------------------------------------------------------------------------------------------------------------------------------------
#cocoAPI

coco_ids_path = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_cocoID_correct.npy'

stimuli_data_path = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_data.npy'
def get_text_features(coco_ids_path , stimuli_data_path ):
    dataDir = '/nfs/diskstation/DataStation/public_dataset/MSCOCO/annotations_trainval2017'
    annFile_trn = '{}/annotations/captions_{}.json'.format(dataDir, 'train2017')
    annFile_val = '{}/annotations/captions_{}.json'.format(dataDir, 'val2017')

    coco_caps_trn = COCO(annFile_trn)
    coco_caps_val = COCO(annFile_val)
    coco_ids = np.load(coco_ids_path)

    #stimulis = torch.tensor(np.load(stimuli_data_path))

    #caps_clip_feature = []
    #caps_LDM_feature_random = []
    caps_LDM_feature_average = []

    for i in tqdm(range(len(coco_ids))):

        coco_id = coco_ids[i]
        annIds = coco_caps_trn.getAnnIds(imgIds=coco_id)
        if len(annIds) == 0:
            annIds = coco_caps_val.getAnnIds(imgIds=coco_id)
            anns = coco_caps_val.loadAnns(annIds)
        else:
            anns = coco_caps_trn.loadAnns(annIds)


        caps = []                                    #每个图像对应5条描述
        for j in range(len(anns)):
            cap = anns[j]['caption']
            caps.append(cap)

        """
        img = stimulis[i,:,:].unsqueeze(0)  #(1,3,256,256)
        img_feature = CLIP_extraction(img).to(device)  #(1,512)
        text = clip.tokenize(caps).to(device)
        text_feature = clip_model.encode_text(text)  #(5,512)

        cos_sim = torch.cosine_similarity(img_feature ,text_feature).cpu().detach().numpy()
        cos_soft = np.exp(cos_sim)/np.sum(np.exp(cos_sim))    
        ind = np.argmax(cos_soft)

        cap_clip_feature = text_feature[ind,:].cpu().detach().numpy()
        caps_clip_feature.append(cap_clip_feature)
        cap_LDM_feature = LDM_model.get_learned_conditioning([caps[ind]]).cpu().detach().numpy().squeeze()
        """

        #idx = random.randint(0,4)
        #cap_LDM_feature_random = LDM_model.get_learned_conditioning([caps[idx]]).cpu().detach().numpy().squeeze()

        cap_LDM_feature_average = 0
        for i in range(5):
            cap = LDM_model.get_learned_conditioning([caps[i]]).cpu().detach().numpy().squeeze()
            cap_LDM_feature_average = cap_LDM_feature_average + cap
        cap_LDM_feature_average = cap_LDM_feature_average/5

        #caps_LDM_feature_random.append(cap_LDM_feature_random)
        caps_LDM_feature_average.append(cap_LDM_feature_average)

    #np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_caps_clip_feature.npy',caps_clip_feature)
    #np.save('./图像重建/数据集/Stable-diffusion文本特征/sub2/val_caps_LDM_feature_len_15_random.npy',caps_LDM_feature_random)
    np.save('./图像重建/数据集/Stable-diffusion文本特征/sub7/trn_caps_LDM_feature_len_15_average.npy',caps_LDM_feature_average)
    return

a = get_text_features(coco_ids_path,stimuli_data_path)
print('')


