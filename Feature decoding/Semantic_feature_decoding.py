from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import fastl2lir
import argparse

ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']

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
def reshape_z(l):  #(8859,15,768)>>(8859,15*768)
	b = []
	for i in range(l.shape[0]):
		a = np.concatenate([l[i,j,:] for j in range(l.shape[1])], axis=0)
		b.append(a)
	b = np.array(b)
	data_, mean, s = z_score(b)
	return data_, mean, s
def cal_pccs(X, Y):
	XMean = np.mean(X)
	YMean = np.mean(Y)
	#标准差
	XSD = np.std(X)
	YSD = np.std(Y)
	#z分数
	ZX = (X-XMean)/(XSD+1e-30)
	ZY = (Y-YMean)/(YSD+1e-30)#相关系数
	r = np.sum(ZX*ZY)/(X.shape[1])           #(len(X))
	return(r)

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


def training_decode_LDM_text_feature(x_trn, y_trn, x_val, y_val, mean, std, model_save_path, n , save):
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
		x = y_val[i, :]*std + mean
		y = pred[i, :]*std + mean
		cor_i = cal_pccs(x, y)
		cor.append(abs(cor_i))
	print("使用{}个体素的预测准确率为{}".format(n,np.mean(cor)))
	return np.mean(cor)

def decode_LDM_text_feature(val_file_ex, n, mean_path, std_path, cls_token_path, model_save_path , recons_img_idx):
    x_test = fetch_ROI_voxel(val_file_ex, ROIs)  # [recons_img_idx:recons_img_idx + 1, :]
    x_test = scaler.fit_transform(x_test)
    mean = (np.load(mean_path)).reshape(1, -1)
    std = (np.load(std_path))
    std[:768] = 0
    std = std.reshape(1, -1)
    model_name = "fastl2_n_feat_{}.pickle".format(n)
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()
    pred = model.predict(x_test)[recons_img_idx:recons_img_idx + 1, :]
    cls = np.load(cls_token_path)

    z = np.zeros((1,11520))
    z[:,:768] = cls
    z[:,768:] = pred
    z_after_reverse = reverse_reshape(z, mean, std)
    return z_after_reverse


def main():
    parser = argparse.ArgumentParser(description='Structural_feature_extraction')
    parser.add_argument('--trn_file_ex', default='', type=str)
    parser.add_argument('--val_file_ex', default='', type=str)
    parser.add_argument('--model_save_path ', default='', type=str)
    parser.add_argument('--CLIP_text_feature_path ', default='', type=str)
    parser.add_argument('--val_CLIP_text_feature_path ', default='', type=str)
    parser.add_argument('--text_mean_path ', default='', type=str)
    parser.add_argument('--text_std_path ', default='', type=str)
    parser.add_argument('--recons_img_idx ', default='', type=float)

    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    text = np.load(args.CLIP_text_feature_path)
    all_data = np.array(torch.tensor(text).squeeze(1))
    _, mean, s = reshape_z(all_data)
    np.save(args.text_mean_path,mean)
    np.save(args.text_std_path,s)
    
    all_data = np.array(torch.tensor(text).squeeze(1))[:, 1:, :]  # (8859,14,768)
    y = _reshape(all_data)
    y_z_score = scaler.fit_transform(y)
    y_trn = y_z_score[:, :]
    y_val = y_z_score[8000:, :]

    x = fetch_ROI_voxel(args.trn_file_ex, ROIs)  # (8859,11694)
    x = scaler.fit_transform(x)
    x_trn = x[:, :]
    x_val = x[8000:, :]

    train = training_decode_LDM_text_feature(x_trn, y_trn, x_val, y_val, mean, s, args.model_save_path, 450 , True)
    
    group = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36,
             0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72,
             0.74, 0.76, 0.78, 0.8]
    cor = []
    y_test = np.load(args.val_CLIP_text_feature_path )
    for i in tqdm(range(982)):
        y_pred = decode_LDM_text_feature(args.val_file_ex, 450, args.mean_path, args.std_path, args.cls_token_path, args.model_save_path , args.recons_img_idx)
        y_pred = _reshape(y_pred)
        y_test_ = y_test[i:i + 1, :, :]
        y_test_ = _reshape(y_test_)
        cor_i = cal_pccs(y_pred, y_test_)
        cor.append(abs(cor_i))

    plt.hist(cor, group, histtype='bar')
    plt.title('15*768')
    plt.show()
    
    


if __name__ == "__main__":
    main()
