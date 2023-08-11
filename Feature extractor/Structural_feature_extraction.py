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
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda:0')
#--------------------------------------------------------------------------------------------------------------------------------
def CLIP_extraction(filepath, target_layers):
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

    data_tensor = torch.tensor(np.load(filepath))
    stimulus_loader = DataLoader(TensorDataset(data_tensor, transform_for_CLIP), batch_size=64)

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
    y=np.concatenate([feature_maps[feature_map] for feature_map in feature_maps ],axis=1)
    data_zscore, mean_all, std_all=z_score(y)
    return data_zscore,mean_all,std_all
    

def main():
    parser = argparse.ArgumentParser(description='Structural_feature_extraction')
    parser.add_argument('--target_layers', help='CLIP_layers', default=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12','VisionTransformer-1'], type=list)
    parser.add_argument('--stim_sub1_trn', help='stim_trn_saved path', default='', type=str)        #(8859,3,256,256)
    parser.add_argument('--stim_sub1_val_multi', help='stim_val_saved path', default='', type=str)  #(982,3,256,256)
    parser.add_argument('--feature_sub1_trn', help='feature_trn_saved root', default='', type=str)
    parser.add_argument('--feature_sub1_val_multi', help='feature_val_saved root', default='', type=str)
    args = parser.parse_args()
    
    stim_sub1_val_CLIP = CLIP_extraction(args.stim_sub1_val_multi, args.target_layers)
    y=np.concatenate([stim_sub1_val_CLIP[feature_map] for feature_map in stim_sub1_val_CLIP ],axis=1)
    data_zscore_val , mean_val , std_val = z_score_all_features(stim_sub1_val_CLIP)
    np.save(args.feature_sub1_val_multi + 'Raw_validation_CLIP.npy', y)
    print("Raw_validation_feature_Done")

    np.save(args.feature_sub1_val_multi + 'val_stim_CLIP_zscore.npy', data_zscore_val)
    np.save(args.feature_sub1_val_multi + 'val_stim_CLIP_mean.npy', mean_val)
    np.save(args.feature_sub1_val_multi + 'val_stim_CLIP_std.npy', std_val)
    print("Z_scored_validation_feature_Done")

    stim_sub1_trn_CLIP = CLIP_extraction(args.stim_sub1_trn, args.target_layers)
    y=np.concatenate([stim_sub1_trn_CLIP[feature_map] for feature_map in stim_sub1_trn_CLIP ],axis=1)
    data_zscore_val , mean_val , std_val = z_score_all_features(stim_sub1_trn_CLIP)
    np.save(args.feature_sub1_trn + 'Raw_trn_CLIP.npy', y)
    print("Raw_trn_feature_Done")

    np.save(args.feature_sub1_trn + 'trn_stim_CLIP_zscore.npy', data_zscore_trn)
    np.save(args.feature_sub1_trn + 'trn_stim_CLIP_mean.npy', mean_trn)
    np.save(args.feature_sub1_trn + 'trn_stim_CLIP_std.npy', std_trn)
    print("Z_scored_trn_feature_Done")


if __name__ == "__main__":
    main()







