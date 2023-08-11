from pycocotools.coco import COCO
import os
import argparse
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
sys.path.append('/data/home/cddu/pythonProject/code/taming-transformers')    #taming-transformer path in your server
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


#------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_text_features(coco_ids_path , stimuli_data_path , feature_saved_path):
    dataDir = '/nfs/diskstation/DataStation/public_dataset/MSCOCO/annotations_trainval2017'
    annFile_trn = '{}/annotations/captions_{}.json'.format(dataDir, 'train2017')
    annFile_val = '{}/annotations/captions_{}.json'.format(dataDir, 'val2017')
    coco_caps_trn = COCO(annFile_trn)
    coco_caps_val = COCO(annFile_val)
    coco_ids = np.load(coco_ids_path)
    caps_LDM_feature_average = []

    for i in tqdm(range(len(coco_ids))):

        coco_id = coco_ids[i]
        annIds = coco_caps_trn.getAnnIds(imgIds=coco_id)
        if len(annIds) == 0:
            annIds = coco_caps_val.getAnnIds(imgIds=coco_id)
            anns = coco_caps_val.loadAnns(annIds)
        else:
            anns = coco_caps_trn.loadAnns(annIds)
            
        caps = []                                   
        for j in range(len(anns)):
            cap = anns[j]['caption']
            caps.append(cap)
            
        cap_LDM_feature_average = 0
        
        for i in range(5):
            cap = LDM_model.get_learned_conditioning([caps[i]]).cpu().detach().numpy().squeeze()
            cap_LDM_feature_average = cap_LDM_feature_average + cap
            
        cap_LDM_feature_average = cap_LDM_feature_average/5
        caps_LDM_feature_average.append(cap_LDM_feature_average)

    np.save(feature_saved_path, caps_LDM_feature_average)
    return

def main():
    parser = argparse.ArgumentParser(description='CLIP text feature extraction')
    parser.add_argument('--stable_diffusion_config_path',  default='', type=str)
    parser.add_argument('--stable_diffusion_skpt_path',  default='', type=str)
    parser.add_argument('--coco_ids_path',  default='', type=str)
    parser.add_argument('--stimuli_data_path',  default='', type=str)
    parser.add_argument('--feature_saved_path',  default='', type=str)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.stable_diffusion_config_path)
    LDM_model = load_model_from_config(config, args.stable_diffusion_skpt_path).to(device)
    a = get_text_features(args.coco_ids_path, args.stimuli_data_path, args.feature_saved_path)


if __name__ == "__main__":
    main()

