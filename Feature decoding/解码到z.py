from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import fastl2lir

#x
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']
trn_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_'
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj07/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'
l = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/sub7/trn_stim_lattent_z.npy')

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value
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

def reverse_reshape(d):
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
#-------------------------------------------------------------------------------------------------------------------------------
model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征/sub7/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

y = reshape(l)
mean = np.mean(y,axis=0).reshape(1, -1)
std = np.std(y,axis=0).reshape(1, -1)
y_z_score = scaler.fit_transform(y)
y_trn = y_z_score[:, :]
y_val = y_z_score[8000:, :]

#x
x = fetch_ROI_voxel(trn_file_ex, ROIs)  # (8859,11694)
x = scaler.fit_transform(x)
x_trn = x[:, :]
x_val = x[8000:, :]

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
		cor_i = np.sum(pred[i,:]*y_val[i,:])/pred.shape[1]
		cor.append(abs(cor_i))
	print("使用{}个体素的预测准确率为{}".format(n,np.mean(cor)))
	return np.mean(cor)

#----------------------------------------------------------------------------------------------------------------------
voxels = [3000]
preds = []
for n in voxels:
	pred = training_decode_LDM_text_feature(n=n , save=True)
	preds.append(pred)
print(preds)
#best_n_index = preds.index(max(preds))
#best_n = voxels[best_n_index]
#a = training_decode_LDM_text_feature(n=best_n  , save=True)
#print("最佳准确率是{}，最佳体素数量是{}".format(a,best_n))

#-----------------------------------------------------------------------------------------------------------------------


def decode_LDM_text_feature(n,recons_img_idx):
    x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
    x_test = scaler.fit_transform(x_test)
    model_save_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/数据集/Stable-diffusion隐空间特征'
    model_name = "仅有视觉区体素fastl2_n_feat_{}.pickle".format(n)
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()
    pred = model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    z_after_reverse = reverse_reshape_z(pred, mean, std)
    return z_after_reverse



