## Code derived from ICGAN Google Colab File

from pytorch_pretrained_biggan import BigGAN, convert_to_images, one_hot_from_names, utils
import sys
import os
sys.path.append('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan')

import torch 

import numpy as np
import torch
import torchvision
import sys
torch.manual_seed(np.random.randint(sys.maxsize))
import imageio
from IPython.display import HTML, Image, clear_output
from PIL import Image as Image_PIL
from scipy.stats import truncnorm, dirichlet
from torch import nn
from base64 import b64encode
from time import time
import warnings
import torchvision.transforms as transforms
import ic_gan.inference.utils as inference_utils
import ic_gan.data_utils.utils as data_utils
from ic_gan.BigGAN_PyTorch.BigGAN import Generator as generator
import sklearn.metrics
import matplotlib.pyplot as plt

def replace_to_inplace_relu(model): #saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)
    return
    
def save(out,name=None, torch_format=True):
  if torch_format:
    with torch.no_grad():
      out = out.cpu().numpy()
  img = convert_to_images(out)[0]
  if name:
    imageio.imwrite(name, np.asarray(img))
  return img


def load_generative_model(gen_model, last_gen_model, experiment_name, model):
  # Load generative model
  if gen_model != last_gen_model:
    model = load_icgan(experiment_name, root_ = './ic_gan')
    last_gen_model = gen_model
  return model, last_gen_model

def load_icgan(experiment_name, root_ = '/content'):
  root = os.path.join(root_, experiment_name)
  config = torch.load("%s/%s.pth" %
                      (root, "state_dict_best0"))['config']

  config["weights_root"] = root_
  config["model_backbone"] = 'biggan'
  config["experiment_name"] = experiment_name
  G, config = inference_utils.load_model_inference(config)
  G.cuda()
  G.eval()
  return G


def get_output(noise_vector, input_label, input_features):  
  if stochastic_truncation: #https://arxiv.org/abs/1702.04782
    with torch.no_grad():
      trunc_indices = noise_vector.abs() > 2*truncation
      size = torch.count_nonzero(trunc_indices).cpu().numpy()
      trunc = truncnorm.rvs(-2*truncation, 2*truncation, size=(1,size)).astype(np.float32)
      noise_vector.data[trunc_indices] = torch.tensor(trunc, requires_grad=requires_grad, device='cuda')
  else:
    noise_vector = noise_vector.clamp(-2*truncation, 2*truncation)
  if input_label is not None:
    input_label = torch.LongTensor(input_label)
  else:
    input_label = None

  out = model(noise_vector, input_label.cuda() if input_label is not None else None, input_features.cuda() if input_features is not None else None)
  
  if channels==1:
    out = out.mean(dim=1, keepdim=True)
    out = out.repeat(1,3,1,1)
  return out

    





def load_feature_extractor(gen_model, last_feature_extractor, feature_extractor):
  # Load feature extractor to obtain instance features
  feat_ext_name = 'classification' if gen_model == 'cc_icgan' else 'selfsupervised'
  if last_feature_extractor != feat_ext_name:
    if feat_ext_name == 'classification':
      feat_ext_path = ''
    else:
      feat_ext_path = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan/pretrained_models/swav_pretrained.pth.tar'
    last_feature_extractor = feat_ext_name
    feature_extractor = data_utils.load_pretrained_feature_extractor(feat_ext_path, feature_extractor = feat_ext_name)
    feature_extractor.eval()
  return feature_extractor, last_feature_extractor


norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
def normalize(image):
  for c in range(3):
    image[:,c] = (image[:,c] - means[c])/stds[c]
  return image

def preprocess_input_image(input_image_path, size): 
  pil_image = Image_PIL.open(input_image_path).convert('RGB')
  transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
  tensor_image = transform_list(pil_image)
  tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
  return tensor_image


def preprocess_generated_image(image):
  
  image = torchvision.transforms.functional.normalize(image*0.5 + 0.5, norm_mean, norm_std)
  image = torch.nn.functional.interpolate(image, 224, mode="bicubic", align_corners=True)
  return image

def imgrid(imarray, cols=4, pad=1, padval=255, row_major=True):
  """Lays out a [N, H, W, C] image array as a single image grid."""
  pad = int(pad)
  if pad < 0:
    raise ValueError('pad must be non-negative')
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=padval)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def image_to_uint8(x):
  """Converts [-1, 1] float array to [0, 255] uint8."""
  x = np.asarray(x)
  x = (256. / 2.) * (x + 1.)
  x = np.clip(x, 0, 255)
  x = x.astype(np.uint8)
  return x