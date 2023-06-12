import os
import ic_gan.BigGAN_PyTorch.utils as biggan_utils
import ic_gan.inference.utils as inference_utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn

def load_generator(exp_name, root_path, backbone, device="cpu"):
	parser = biggan_utils.prepare_parser()
	parser = biggan_utils.add_sample_parser(parser)
	parser = inference_utils.add_backbone_parser(parser)

	args = ["--experiment_name", exp_name]
	args += ["--base_root", root_path]
	args += ["--model_backbone", backbone]

	config = vars(parser.parse_args(args=args))

	# Load model and overwrite configuration parameters if stored in the model
	config = biggan_utils.update_config_roots(config, change_weight_folder=False)
	generator, config = inference_utils.load_model_inference(config, device=device)
	biggan_utils.count_parameters(generator)
	generator.eval()
	return generator
generator = load_generator('icgan_biggan_imagenet_res256',
                           './图像重建/对照实验/Cortex2Image/ic_gan/pretrained_models', 'biggan')
print('生成器加载完毕')
