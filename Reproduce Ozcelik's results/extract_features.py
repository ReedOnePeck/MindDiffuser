from model_utils import load_feature_extractor, preprocess_generated_image
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from PIL import Image
import numpy as np
import h5py
import torch
import sys
import os

class batch_generator_external_images(Dataset):
    """
    Generates batches of images from a directory
    :param img_size: image should be resized to size
    :param batch_size: batch size
    :param ext_dir: directory containing images
    """
    def __init__(self, data_path = './图像重建/对照实验/ic-gan-recons/stims/images_256.npz', mode='test_images'):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.mode = mode
        self.data = self.data[self.mode].astype(np.float32)


    def __getitem__(self,idx):
        img = self.data[idx]
        img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.data)


feature_extractor_name = 'selfsupervised'
gen_model = 'icgan'
last_feature_extractor = None
feature_extractor = None
# Load feature extractor (outlier filtering and optionally input image feature extraction)
feature_extractor, last_feature_extractor = load_feature_extractor(gen_model, last_feature_extractor, feature_extractor)

eps = 1e-8


image_path = './图像重建/对照实验/ic-gan-recons/stims/images_256.npz'
batch_size = 5
train_images = batch_generator_external_images(data_path = image_path, mode='train_images')
trainloader = DataLoader(train_images,batch_size,shuffle=False)

test_images = batch_generator_external_images(data_path = image_path,mode='test_images')
testloader = DataLoader(test_images,batch_size,shuffle=False)

num_features, num_test, num_train = 2048, len(test_images), len(train_images)
train_instance = np.zeros((num_train,num_features))
test_instance = np.zeros((num_test,num_features))


for i,batch in enumerate(testloader):

  num_sample = len(batch)
  images_batch = torch.tensor(batch).permute(0,3,1,2).cuda()
  images_batch = preprocess_generated_image(images_batch)
  input_features, _ = feature_extractor(images_batch)
  input_features/=torch.linalg.norm(input_features.clone().detach(),dim=-1, keepdims=True)
  input_features = input_features.detach().cpu().numpy()
  test_instance[i*batch_size: (i+1)*batch_size] = input_features
  print(i*batch_size)

for i,batch in enumerate(trainloader):
 
  num_sample = len(batch)
  images_batch = torch.tensor(batch).permute(0,3,1,2).cuda()
  images_batch = preprocess_generated_image(images_batch)
  input_features, _ = feature_extractor(images_batch)
  input_features/=torch.linalg.norm(input_features.clone().detach(),dim=-1, keepdims=True)
  input_features = input_features.detach().cpu().numpy()
  train_instance[i*batch_size: (i+1)*batch_size] = input_features
  print(i*batch_size)

np.savez('./图像重建/对照实验/ic-gan-recons/stims/extracted_features/instance_features.npz', train_instance=train_instance, test_instance=test_instance)
