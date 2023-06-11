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


#二、CLIP文本特征
#其实可以不要这一部分，因为之前做重建天花板的时候就没有考虑拿CLIP文本特征做约束



#三、LDM自带的文本编码器的文本特征






#-------------------------------------------------------------------------------------------------------------------------------------------------------

"""
from scipy.io import loadmat
exp_design = loadmat("/nfs/diskstation/DataStation/public_dataset/NSD/nsddata/experiments/nsd/nsd_expdesign.mat")
ordering = exp_design['masterordering'].flatten() - 1  #ordering中存放10000张图像在刺激时呈现的顺序
data_size = 27750                                      #被试1实际测量出的刺激-fMRI信号对
ordering_data = ordering[:data_size]
shared_mask = ordering_data<1000                       #10000张图像中，前1000张是所有被试都会看的，这些fMRI数据作为验证集
b = ordering_data[shared_mask]
idx, idx_count = np.unique(ordering_data, return_counts=True)
idx_list = [ordering_data==i for i in idx]
shared_mask_mt = idx<1000
c = idx[shared_mask_mt]
"""
"""
对NSD数据集中sub1数据的描述：
（1）被试1的刺激数据集一共是10000张图像（前1000张对应的fMRI作为val_set，后9000张对应trn_set），总共呈现3次，即30000次刺激，呈现顺序在ordering_data中，但是只测得了27750组刺激-fMRI对。
对这些测得的数据，按照刺激索引平均后，得到9841组训练集数据，982组验证集数据
（2）这10000张图像在COCO中的索引在/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_stimuli/stimuli/nsd/nsd_stim_info_merged.csv中查找
但是由于这10000张图像在csv文件中是乱序的，没办法直接索引验证集中图像对应的COCO索引，因此后续在重建验证集图像的时候，首先，在csv文件中把shared图像及其COCO索引挑出来，然后从COCO中索引出图像和验证集的图像一一对应（肉眼观察）
"""
