from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
#from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import fastl2lir
import numpy as np
import sys
sys.path.append('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan')
import ic_gan.inference.utils as inference_utils
from torch import nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda:0')


#x
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']  #
trn_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_'
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'

def z_score(data):
    scaler = StandardScaler()
    data_=scaler.fit_transform(data)
    mean=np.mean(data,axis=0)
    s=np.std(data,axis=0)
    return data_,mean,s

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

def cal_pccs(X, Y):
	XMean = np.mean(X)
	YMean = np.mean(Y)
	#标准差
	XSD = np.std(X)
	YSD = np.std(Y)
	#z分数
	ZX = (X-XMean)/(XSD+1e-30)
	ZY = (Y-YMean)/(YSD+1e-30)#相关系数
	r = np.sum(ZX*ZY)/(2048)           #(len(X))
	return(r)

x = fetch_ROI_voxel(trn_file_ex, ROIs)  # (8859,11694)
x = scaler.fit_transform(x)
x_trn = x[:8000, :]
x_val = x[8000:, :]

y = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/stims/extracted_features/instance_features.npz')['train_instance']
data_,mean,s = z_score(y)
np.save('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/mean.py',mean)
np.save('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/std.py',s)

y_trn = data_[:8000, :]
y_val = data_[8000:, :]

model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/results/解码模型保存/'

def training_decode_LDM_text_feature(n , save):
	model = fastl2lir.FastL2LiR()
	model.fit(x_trn, y_trn, alpha=0.15, n_feat=n)
	pred = model.predict(x_val)

	if save:
		model_name = "fastl2_n_feat_{}.pickle".format(n)
		f = open(model_save_path + model_name, 'wb')
		pickle.dump(model, f, protocol=4)
		f.close()

	cor = []
	for i in range(pred.shape[0]):
		x = y_val[i, :]*s + mean
		y = pred[i, :]*s + mean
		cor_i = cal_pccs(x, y)
		cor.append(abs(cor_i))
	print("使用{}个体素的预测准确率为{}".format(n,np.mean(cor)))
	return np.mean(cor)

voxels = [200,250,300,350,400,500]
preds = []
for n in voxels:
	pred = training_decode_LDM_text_feature(n=n , save=True)
	preds.append(pred)
	print("体素数量{},准确率{}".format(n,pred))

print(preds)










