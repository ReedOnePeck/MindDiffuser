from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import fastl2lir
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
scaler = StandardScaler()
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']
group = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34,0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,0.72, 0.74, 0.76, 0.78, 0.8]

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
	XSD = np.std(X)
	YSD = np.std(Y)
	ZX = (X-XMean)/(XSD+1e-30)
	ZY = (Y-YMean)/(YSD+1e-30)
	r = np.sum(ZX*ZY)/(len(X))
	return(r)
	
def decoding_layer(trn_CLIP_feature_path, val_CLIP_feature_path, mask_path, layer_name, remain_rate, decoding_function, n, model_save_path ):
    model_save_path = model_save_path + '{}_{}/'.format(layer_name, remain_rate)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    x_trn = fetch_ROI_voxel(trn_file_ex, ROIs)  # (8859,11694)
    x_mean = np.mean(x_trn, axis=0)
    x_std = np.std(x_trn, axis=0)

    x_trn = (x_trn - x_mean) / x_std
    trn_CLIP_feature_all_layer = fetch_CLIP_feature(trn_CLIP_feature_path + '/trn_stim_CLIP_zscore.npy')

    x_val = fetch_ROI_voxel(val_file_ex, ROIs)[:500, :]
    x_val = (x_val - x_mean) / x_std
    val_CLIP_feature_all_layer = fetch_CLIP_feature(val_CLIP_feature_path + '/val_stim_CLIP_zscore.npy')

    if layer_name != 'VisionTransformer-1':
        mask = np.load(mask_path + '/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
        y_trn = CLIP_feature_layer(trn_CLIP_feature_all_layer, layer_name)[:8000,mask]
        y_val = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[8000:, mask]
    else:
        y_trn = CLIP_feature_layer(trn_CLIP_feature_all_layer, layer_name)[:8000,:]
        y_val = CLIP_feature_layer(val_CLIP_feature_all_layer, layer_name)[8000:, :]
    print('数据加载完毕')


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
        x = y_val[:, i]
        y = pred[:, i]
        cor_i = cal_pccs(x, y)
        cor.append(abs(cor_i))

    mean_cor = np.mean(np.array(cor))
    print("{}_{}:".format(layer_name,model_name), mean_cor)
    plt.hist(cor, group, histtype='bar')
    plt.title('val_{}_{}'.format(layer_name,model_name))
    plt.show()
    return


def decoding_test_1layer(model_save_path, mask_path, layer_name, remain_rate, decoding_function, n):
    if decoding_function == 'fastl2':
        model_name = "fastl2_n_feat_{}.pickle".format(n)
    else:
        model_name = "himalaya.pickle"
    f_save = open(model_save_path + model_name, 'rb')
    model = pickle.load(f_save)
    f_save.close()

    if layer_name != 'VisionTransformer-1':
        mask = np.load(mask_path + '/{}/Folder_5_{}.npy'.format(layer_name,remain_rate))
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
    return pred  



def main():
    parser = argparse.ArgumentParser(description='Structural_feature_extraction')
    parser.add_argument('--trn_file_ex', default='', type=str)
    parser.add_argument('--val_file_ex', default='', type=str)
    parser.add_argument('--model_save_path ', default='', type=str)
    parser.add_argument('--trn_CLIP_feature_path', default='', type=str)
    parser.add_argument('--val_CLIP_feature_path', default='', type=str)
    parser.add_argument('--mask_path', default='', type=str)
    parser.add_argument('--Layers', default=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12','VisionTransformer-1'], type=list)
    parser.add_argument('--recons_img_idx ', default='', type=float)
    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    for layer in args.Layers:
	    b = decoding_layer(trn_CLIP_feature_path=args.trn_CLIP_feature_path, val_CLIP_feature_path=args.val_CLIP_feature_path, mask_path = args.mask_path, 
			       layer_name=layer, remain_rate=25, decoding_function='fastl2', n=850, model_save_path=args.model_save_path )

if __name__ == "__main__":
    main()
	






