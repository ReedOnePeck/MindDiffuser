import numpy as np
import sys
sys.path.append('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan')
import ic_gan.inference.utils as inference_utils
from torch import nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
instance = np.load('/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/stims/extracted_features/instance_features.npz')['test_instance']
i = torch.tensor(instance[10:11].astype(np.float32)).to(device)
n = torch.tensor(np.random.randn(1,119).astype(np.float32)).to(device)


def replace_to_inplace_relu(
        model):  # saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
  for child_name, child in model.named_children():
    if isinstance(child, nn.ReLU):
      setattr(model, child_name, nn.ReLU(inplace=False))
    else:
      replace_to_inplace_relu(child)
  return

def load_icgan(experiment_name, root_ = '/nfs/diskstation/DataStation/ChangdeDu/LYZ/图像重建/对照实验/ic-gan-recons/ic_gan/pretrained_models'):
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

experiment_name = 'icgan_biggan_imagenet_res256'

model = load_icgan(experiment_name)
replace_to_inplace_relu(model)
eps = 1e-8

print('Model is loaded')

out = model(n,  None, i)
print(out)







