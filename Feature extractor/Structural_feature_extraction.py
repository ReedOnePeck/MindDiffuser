#https://github.com/styvesg/nsd/blob/master/torched_alexnet_fwrf.ipynb
#https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from packages import model_options as MO
from packages import feature_extraction_detach as FE

#from himalaya.lasso import SparseGroupLassoCV

#from himalaya.backend import set_backend
#backend = set_backend("torch_cuda", on_error="warn")

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda:0')
#--------------------------------------------------------------------------------------------------------------------------------
#提取训练集和验证集图像刺激的CLIP特征
def CLIP_extraction(filepath):
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

    data_tensor = torch.tensor(np.load(filepath))#[8000:,:,:]
    stimulus_loader = DataLoader(TensorDataset(data_tensor, transform_for_CLIP), batch_size=64)
    target_layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12','VisionTransformer-1']
    feature_maps = FE.get_all_feature_maps(model, stimulus_loader, layers_to_retain=target_layers,
                                           remove_duplicates=False, numpy=True)

    for feature_map in feature_maps:
        print(feature_map, feature_maps[feature_map].shape)

    return feature_maps

def z_score(data):
    scaler = StandardScaler()
    data_=scaler.fit_transform(data)
    mean=np.mean(data,axis=0)
    s=np.std(data,axis=0)
    return data_,mean,s

def z_score_all_features(feature_maps):
    """
    对所有的CLIP提出来的特征进行标准化，并且记录下没他们的均值与标准差方便后续的反变换
    :param feature_maps: CLIP提出来的特征
    :return:
    """
    y=np.concatenate([feature_maps[feature_map] for feature_map in feature_maps ],axis=1)
    data_zscore, mean_all, std_all=z_score(y)
    return data_zscore,mean_all,std_all
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#处理数据集文本编码文本编码
def make_texts(file_path):
    """
    :param file_path: 存储训练集/验证集刺激的文本路径
    :return: 把文本提取出来
    """

    return

#提取LDM文本编码器的特征
def text_feature_LDM(texts):
    return





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(8859,2392)
#sub1_trn = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_V1.npy'
#(982,2392)
#sub1_val_multi = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_V1.npy'

#(8859,3,256,256)
stim_sub1_trn = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_data.npy'
#(982,3,256,256)
stim_sub1_val_multi = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_multi_trial_data.npy'
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#数据处理与存储
#一、CLIP视觉特征

"""
stim_sub1_val_CLIP = CLIP_extraction(stim_sub1_val_multi)
y=np.concatenate([stim_sub1_val_CLIP[feature_map] for feature_map in stim_sub1_val_CLIP ],axis=1)
data_zscore_val , mean_val , std_val = z_score_all_features(stim_sub1_val_CLIP)
print("验证集数据提取完毕")
print(mean_val)
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP.npy', y)


#(984,38400*12+512)  存储的是zcore后的训练集存储数据，其中CLIP的特征只提取了2-24层的偶数层特征。以及最后的512维的语义特征
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP_zscore.npy', data_zscore_val)
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP_mean.npy', mean_val)
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP_std.npy', std_val)
print("验证集数据保存完毕")
"""

stim_sub1_trn_CLIP = CLIP_extraction(stim_sub1_val_multi)
y=np.concatenate([stim_sub1_trn_CLIP[feature_map] for feature_map in stim_sub1_trn_CLIP ],axis=1)
data_zscore_trn , mean_trn , std_trn = z_score_all_features(stim_sub1_trn_CLIP)
print("训练集数据提取完毕")
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_8.npy', y)



#(8859,38400*12+512)  存储的是zcore后的训练集存储数据，其中CLIP的特征只提取了2-24层的偶数层特征。以及最后的512维的语义特征
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_zscore_8.npy', data_zscore_trn)
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_mean_8.npy', mean_trn)
np.save('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_std_8.npy', std_trn)
print("训练集数据保存完毕")




