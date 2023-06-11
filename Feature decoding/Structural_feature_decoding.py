from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
#from sklearn.model_selection import KFold
#import torch
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import fastl2lir
#from himalaya.lasso import SparseGroupLassoCV
#from himalaya.backend import set_backend
#backend = set_backend("torch_cuda", on_error="warn")


ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']
trn_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_voxel_data_'
val_file_ex = '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_voxel_multi_trial_data_'
def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

def fetch_CLIP_feature(file_name):
    CLIP_feature_all_layer = np.load(file_name)
    return CLIP_feature_all_layer

def CLIP_feature_layer(CLIP_feature_all_layer,layer_name):
    if layer_name == 'VisionTransformer-1':
        CLIP_layer = CLIP_feature_all_layer[:,38400*6:]
    elif layer_name[-2] == '-':
        index = int(int(layer_name[-1])/2)
        CLIP_layer = CLIP_feature_all_layer[:, 38400 * (index - 1):38400 * index]
    else:
        index = int(int(layer_name[-2:])/2)
        CLIP_layer = CLIP_feature_all_layer[:, 38400 * (index - 1):38400 * index]
    return CLIP_layer

def cal_pccs(X, Y):
	XMean = np.mean(X)
	YMean = np.mean(Y)
	#标准差
	XSD = np.std(X)
	YSD = np.std(Y)
	#z分数
	ZX = (X-XMean)/(XSD+1e-30)
	ZY = (Y-YMean)/(YSD+1e-30)#相关系数
	r = np.sum(ZX*ZY)/(len(X))
	return(r)
group = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34,0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,0.72, 0.74, 0.76, 0.78, 0.8]



def decoding_layer(layer_name,remain_rate,decoding_function,n):
    model_save_path = './图像重建/数据集/解码特征选取/{}/解码模型权重保存/{}_remain_{}/'.format(layer_name,decoding_function,remain_rate)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


    x_trn = fetch_ROI_voxel(trn_file_ex, ROIs)  # (8859,11694)
    x_mean = np.mean(x_trn, axis=0)
    x_std = np.std(x_trn, axis=0)

    x_trn = (x_trn - x_mean) / x_std
    trn_CLIP_feature_all_layer = np.concatenate([fetch_CLIP_feature(
        "/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/trn_stim_CLIP_zscore_{}.npy".format(
            i + 1)) for i in range(8)], axis=0)

    x_val = fetch_ROI_voxel(val_file_ex, ROIs)[:500, :]
    x_val = (x_val - x_mean) / x_std

    # x_val = scaler.fit_transform(x_val)
    val_CLIP_feature_all_layer = fetch_CLIP_feature(
        '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_CLIP_zscore.npy')

    if layer_name != 'VisionTransformer-1':
        mask = np.load('./图像重建/数据集/解码特征选取/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
        y_trn = CLIP_feature_layer(trn_CLIP_feature_all_layer, layer_name)[:,mask]
        y_val = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[:500, mask]
    else:
        y_trn = CLIP_feature_layer(trn_CLIP_feature_all_layer, layer_name)
        y_val = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[:500, :]

    print('数据加载完毕')

    """
    y_means = ((np.mean(np.array([np.load(
        '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_mean_{}.npy'.format(
            i + 1)) for i in range(8)]), axis=0))[:38400 ])[mask]

    y_stds = ((np.mean(np.array([np.load(
        '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/trn_stim_CLIP_std_{}.npy'.format(
            i + 1)) for i in range(8)]), axis=0))[:38400 ])[mask]


    y_val_mean = (np.load(
        '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP_mean.npy')[
                 0:38400 ])[mask]

    y_val_std = (np.load(
        '/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj02/func1pt8mm/betas_fithrf_GLMdenoise_RR/部分CLIP特征/val_stim_CLIP_std.npy')[
                0:38400])[mask]
    """

    if decoding_function == 'fastl2':
        model = fastl2lir.FastL2LiR()
        model.fit(x_trn, y_trn, alpha=0.15, n_feat=n)   # alpha=0.15
        pred = model.predict(x_val)

        model_name ="fastl2_n_feat_{}.pickle".format(n)
        f = open(model_save_path+model_name,'wb')
        pickle.dump(model, f)
        f.close()

    cor = []
    for i in tqdm(range(pred.shape[1])):
        x = y_val[:, i]#(y_val*y_val_std+y_val_mean)[:, i]
        y = pred[:, i]#(pred*y_stds+y_means)[:, i]
        cor_i = cal_pccs(x, y)
        cor.append(abs(cor_i))

    mean_cor = np.mean(np.array(cor))
    print("{}_{}:".format(layer_name,model_name), mean_cor)
    plt.hist(cor, group, histtype='bar')
    plt.title('val_{}_{}'.format(layer_name,model_name))
    plt.show()
    return



b = decoding_layer('VisionTransformer-1', 100, 'fastl2', 450)
print('')

"""
for i in range(6):
    layer = 'Linear-{}'.format((i+1)*2)
    if (i+1)*2 <= 6:
        b = decoding_layer(layer, 25, 'fastl2', 350)
        #predicted2 = decoding_test_1layer(layer, 75, 'fastl2', 350)
    else:
        b = decoding_layer(layer, 25, 'fastl2', 250)
        #predicted2 = decoding_test_1layer(layer, 75, 'fastl2', 250)
"""















x_val = fetch_ROI_voxel(val_file_ex, ROIs)[500:, :]
x_test = scaler.fit_transform(x_val)
val_CLIP_feature_all_layer = fetch_CLIP_feature('/nfs/diskstation/DataStation/public_dataset/NSD/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/val_stim_CLIP_zscore.npy')

def decoding_test_1layer(layer_name,remain_rate,decoding_function,n):
    model_save_path = './图像重建/数据集/解码特征选取/{}/解码模型权重保存/{}_remain_{}/'.format(layer_name,decoding_function,remain_rate)

    if decoding_function == 'fastl2':
        model_name = "fastl2_n_feat_{}.pickle".format(n)
    else:
        model_name = "himalaya.pickle"
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()

    if layer_name != 'VisionTransformer-1':
        mask = np.load('./图像重建/数据集/解码特征选取/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
        y_test = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[500:, mask]
    else:
        y_test = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[500:, :]

    if decoding_function == 'fastl2':
        pred = model.predict(x_test)
    else:
        pred = model.predict(x_test).cpu().detach().numpy()

    cor = []
    for i in tqdm(range(pred.shape[1])):
        x = y_test[:, i]
        y = pred[:, i]
        cor_i = cal_pccs(x, y)
        cor.append(abs(cor_i))
    plt.hist(cor, group, histtype='bar')
    plt.title('test_{}_{}'.format(layer_name,model_name))
    plt.show()
    mean_cor = np.mean(np.array(cor))
    print("{}_{}:".format(layer_name,model_name),mean_cor)
    return pred  #这是zscore后的，别忘了乘以标准差，加均值


"""

b = decoding_layer('Linear-2', 5, 'fastl2', 1050)

c = decoding_layer('Linear-4', 5, 'fastl2', 850)
c1 = decoding_layer('Linear-4', 5, 'fastl2', 150)
c2 = decoding_layer('Linear-4', 5, 'fastl2', 1050)

d = decoding_layer('Linear-6', 5, 'fastl2', 850)
d1 = decoding_layer('Linear-6', 5, 'fastl2', 150)
d2 = decoding_layer('Linear-6', 5, 'fastl2', 1050)

e = decoding_layer('Linear-8', 5, 'fastl2', 850)
e1 = decoding_layer('Linear-8', 5, 'fastl2', 150)
e2 = decoding_layer('Linear-8', 5, 'fastl2', 1050)
"""





#predicted2 = decoding_test_1layer('Linear-8', 25, 'fastl2', 250)













