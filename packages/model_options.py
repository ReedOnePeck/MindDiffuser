import pandas as pd
import numpy as np
import os, sys, torch
import importlib.util
from warnings import filterwarnings

model_typology = pd.read_csv( '/nfs/diskstation/DataStation/ChangdeDu/luyizhuo/model_typology.csv')
model_typology['model_name'] = model_typology['model']
model_typology['model_type'] = model_typology['model_type'].str.lower()

def subset_typology(model_source):
    return model_typology[model_typology['model_source'] == model_source].copy()

# Torchvision Options ---------------------------------------------------------------------------

from torch.hub import load_state_dict_from_url
import torchvision.models as models

def define_torchvision_options():
    torchvision_options = {}

    model_types = ['imagenet','inception','segmentation', 'detection', 'video']
    pytorch_dirs = dict(zip(model_types, ['.','.','.segmentation.', '.detection.', '.video.']))

    torchvision_typology = model_typology[model_typology['model_source'] == 'torchvision'].copy()
    torchvision_typology['model_type'] = torchvision_typology['model_type'].str.lower()
    training_calls = {'random': '(pretrained=False)', 'pretrained': '(pretrained=True)'}
    for index, row in torchvision_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        model_source = 'torchvision'
        for training in ['random', 'pretrained']:
            train_type = row['model_type'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            model_call = 'models' + pytorch_dirs[model_type] + model_name + training_calls[training]
            torchvision_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                                  'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return torchvision_options

import torchvision.transforms as transforms

def get_torchvision_transforms(model_type, input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}

    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]

    if model_type == 'imagenet':
        specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]

    if model_type == 'segmentation':
        specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]

    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms

    return transforms.Compose(recommended_transforms)


# Taskonomy Options ---------------------------------------------------------------------------


def retrieve_taskonomy_encoder(model_name, verbose = False):
    from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS
    from visualpriors import taskonomy_network

    weights_url = TASKONOMY_PRETRAINED_URLS[model_name + '_encoder']
    weights = torch.utils.model_zoo.load_url(weights_url)
    if verbose: print('{} weights loaded succesfully.'.format(model_name))
    model = taskonomy_network.TaskonomyEncoder()
    model.load_state_dict(weights['state_dict'])

    return model

def define_taskonomy_options():
    taskonomy_options = {}

    task_typology = model_typology[model_typology['model_source'] == 'taskonomy'].copy()
    for index, row in task_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type, model_source = 'taskonomy', 'taskonomy'
        model_string = model_name + '_' + train_type
        model_call = "retrieve_taskonomy_encoder('{}')".format(model_name)
        taskonomy_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                            'train_type': train_type, 'model_source': model_source, 'call': model_call})

    taskonomy_options['random_weights_taskonomy'] = ({'model_name': 'random_weights', 'model_type': 'taskonomy',
                                                      'train_type': 'taskonomy', 'model_source': 'taskonomy',
                                                      'call': 'taskonomy_network.TaskonomyEncoder()'})

    return taskonomy_options

import torchvision.transforms.functional as functional

def taskonomy_transform(image):
    return (functional.to_tensor(functional.resize(image, (256,256))) * 2 - 1)#.unsqueeze_(0)

def get_taskonomy_transforms(input_type = 'PIL'):
    recommended_transforms = taskonomy_transform
    if input_type == 'PIL':
        return recommended_transforms
    if input_type == 'numpy':
        def functional_from_numpy(image):
            image = functional.to_pil_image(image)
            return recommended_transforms(image)
        return functional_from_numpy

# Timm Options ---------------------------------------------------------------------------

def retrieve_timm_model(model_name, pretrained = True):
    from timm import create_model
    return create_model(model_name, pretrained)

def define_timm_options():
    timm_options = {}

    timm_typology = model_typology[model_typology['model_source'] == 'timm'].copy()
    for index, row in timm_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        model_source = 'timm'
        for training in ['random', 'pretrained']:
            train_type = row['model_type'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            train_bool = False if training == 'random' else True
            model_call = "retrieve_timm_model('{}', pretrained = {})".format(model_name, train_bool)
            timm_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                           'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return timm_options

def modify_timm_transform(timm_transform):

    transform_list = timm_transform.transforms

    crop_index, crop_size = next((index, transform.size) for index, transform
                             in enumerate(transform_list) if 'CenterCrop' in str(transform))
    resize_index, resize_size = next((index, transform.size) for index, transform
                                     in enumerate(transform_list) if 'Resize' in str(transform))

    transform_list[resize_index].size = crop_size
    transform_list.pop(crop_index)
    return transforms.Compose(transform_list)

def get_timm_transforms(model_name, input_type = 'PIL'):
    from timm.data.transforms_factory import create_transform
    from timm.data import resolve_data_config

    config = resolve_data_config({}, model = model_name)
    timm_transforms = create_transform(**config)
    timm_transform = modify_timm_transform(timm_transforms)

    if input_type == 'PIL':
        recommended_transforms = timm_transform.transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + timm_transform.transforms

    return transforms.Compose(recommended_transforms)

# CLIP Options ---------------------------------------------------------------------------
device = torch.device('cuda:0')
def retrieve_clip_model(model_name):
    # 这里CPU改了GPU
    import clip; model, _ = clip.load(model_name, device=device)
    return model.visual

def define_clip_options():
    clip_options = {}

    clip_typology = model_typology[model_typology['model_source'] == 'clip'].copy()
    for index, row in clip_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type = row['train_type']
        model_source = 'clip'
        model_string = '_'.join([model_name, train_type])
        model_call = "retrieve_clip_model('{}')".format(model_name)
        clip_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                       'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return clip_options

def get_clip_transforms(model_name, input_type = 'PIL'):
    # 这里CPU改了GPU
    import clip; _, preprocess = clip.load(model_name, device = device)
    if input_type == 'PIL':
        recommended_transforms = preprocess.transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + preprocess.transforms
    recommended_transforms = transforms.Compose(recommended_transforms)
    if 'ViT' in model_name:
        def transform_plus_retype(image_input):
            return recommended_transforms(image_input).type(torch.HalfTensor)
        return transform_plus_retype
    if 'ViT' not in model_name:
        return recommended_transforms

# VISSL Options ---------------------------------------------------------------------------

def retrieve_vissl_model(model_name):
    vissl_data = (model_typology[model_typology['model_source'] == 'vissl']
                  .set_index('model_name').to_dict('index'))
#这里CPU改了GPU
    weights = load_state_dict_from_url(vissl_data[model_name]['weights_url'], map_location = torch.device('cuda:0'))

    def replace_module_prefix(state_dict, prefix, replace_with = ''):
        return {(key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
                      for (key, val) in state_dict.items()}

    def convert_model_weights(model):
        if "classy_state_dict" in model.keys():
            model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in model.keys():
            model_trunk = model["model_state_dict"]
        else:
            model_trunk = model
        return replace_module_prefix(model_trunk, "_feature_blocks.")

    converted_weights = convert_model_weights(weights)
    excess_weights = ['fc','projection', 'prototypes']
    converted_weights = {key:value for (key,value) in converted_weights.items()
                             if not any([x in key for x in excess_weights])}

    if 'module' in next(iter(converted_weights)):
        converted_weights = {key.replace('module.',''):value for (key,value) in converted_weights.items()
                             if 'fc' not in key}

    from torchvision.models import resnet50
    import torch.nn as nn

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model = resnet50()
    model.fc = Identity()

    model.load_state_dict(converted_weights)

    return model


def define_vissl_options():
    vissl_options = {}

    vissl_typology = model_typology[model_typology['model_source'] == 'vissl'].copy()
    for index, row in vissl_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type = row['train_type']
        model_source = 'vissl'
        model_string = '_'.join([model_name, train_type])
        model_call = "retrieve_vissl_model('{}')".format(model_name)
        vissl_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                        'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return vissl_options


def get_vissl_transforms(input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}

    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]
    specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]

    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms

    return transforms.Compose(recommended_transforms)

# Dino Options ---------------------------------------------------------------------------

def define_dino_options():
    dino_options = {}

    dino_typology = model_typology[model_typology['model_source'] == 'dino'].copy()
    for index, row in dino_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type = row['train_type']
        model_source = 'dino'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('facebookresearch/dino:main', '{}')".format(model_name)
        dino_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                       'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return dino_options

def get_dino_transforms(model_type, input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}

    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]

    specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]

    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms

    return transforms.Compose(recommended_transforms)


# MiDas Options ---------------------------------------------------------------------------

def define_midas_options():
    midas_options = {}

    midas_typology = model_typology[model_typology['model_source'] == 'midas'].copy()
    for index, row in midas_typology.iterrows():
        model_name = row['model']
        model_type = row['model_type']
        train_type = row['train_type']
        model_source = 'midas'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('intel-isl/MiDaS','{}')".format(model_name)
        midas_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                        'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return midas_options

def get_midas_transforms(model_name, input_type = 'PIL'):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_name in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    if model_name not in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.small_transform

    if input_type == 'PIL':
        recommended_transforms = [lambda img: np.array(img)] + transform.transforms
    if input_type == 'numpy':
        recommended_transforms = transform.transforms

    return transforms.Compose(recommended_transforms)

# Detectron2 Options ---------------------------------------------------------------------------
"""

def retrieve_detectron_model(model_name, backbone_only = True):
    from detectron2.modeling import build_model
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer

    detectron_data = subset_typology('detectron')
    detectron_dict = (detectron_data.set_index('model').to_dict('index'))
    weights_path = detectron_dict[model_name]['weights_url']

    cfg = model_zoo.get_config(weights_path)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_path)

    cfg_clone = cfg.clone()
    model = build_model(cfg_clone)
    model = model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    if backbone_only:
        return model.backbone
    if not backbone_only:
        return model


def define_detectron_options():
    detectron_options = {}

    detectron_typology = model_typology[model_typology['model_source'] == 'detectron'].copy()
    for index, row in detectron_typology.iterrows():
        model_name = row['model']
        model_type = row['model_type']
        train_type = row['train_type']
        model_source = 'detectron'
        model_string = '_'.join([model_name, train_type])
        model_call = "retrieve_detectron_model('{}')".format(model_name)
        detectron_options[model_string] = ({'model_name': model_name, 'model_type': model_type,
                                            'train_type': train_type, 'model_source': model_source, 'call': model_call})

    return detectron_options

def get_detectron_transforms(model_name, input_type = 'PIL'):
    import detectron2.data.transforms as detectron_transform
    from detectron2 import model_zoo

    detectron_data = subset_typology('detectron')
    detectron_dict = (detectron_data.set_index('model').to_dict('index'))
    weights_path = detectron_dict[model_name]['weights_url']

    cfg = model_zoo.get_config(weights_path)
    model = retrieve_detectron_model(model_name, backbone_only = False)

    augment = detectron_transform.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST,
                                                      cfg.INPUT.MIN_SIZE_TEST],
                                                      cfg.INPUT.MAX_SIZE_TEST)

    def detectron_transforms(original_image):
        if input_type == 'PIL':
            original_image = np.asarray(original_image)
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = augment.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        return model.preprocess_image([inputs]).tensor

    return detectron_transforms
"""
# Aggregate Options ---------------------------------------------------------------------------

def get_model_options(model_type = None, train_type=None, train_data = None, model_source=None):
    model_options = {**define_torchvision_options(), **define_taskonomy_options(),
                     **define_timm_options(),  **define_clip_options(),
                     **define_vissl_options(), **define_dino_options(),
                     **define_midas_options()}

    if model_type is not None:
        model_options = {string: info for (string, info) in model_options.items()
                         if model_options[string]['model_type'] in model_type}

    if train_type is not None:
        model_options = {string: info for (string, info) in model_options.items()
                         if model_options[string]['train_type'] in train_type}

    if model_source is not None:
        model_options = {string: info for (string, info) in model_options.items()
                         if model_options[string]['model_source'] in model_source}

    return model_options


transform_options = {'torchvision': get_torchvision_transforms,
                     'timm': get_timm_transforms,
                     'taskonomy': get_taskonomy_transforms,
                     'clip': get_clip_transforms,
                     'vissl': get_vissl_transforms,
                     'dino': get_dino_transforms,
                     'midas': get_midas_transforms,
                     }

def get_transform_options():
    return transform_options

def get_recommended_transforms(model_query, input_type = 'PIL'):
    cached_model_types = ['imagenet','taskonomy','vissl']
    model_types = model_typology['model_type'].unique()
    if model_query in get_model_options():
        model_option = get_model_options()[model_query]
        model_type = model_option['model_type']
        model_name = model_option['model_name']
        model_source = model_option['model_source']
    if model_query in model_types:
        model_type = model_query
    if model_query not in list(get_model_options()) + list(model_types):
        raise ValueError('Query is neither a model_string nor a model_type.')

    if model_type in cached_model_types:
        if model_type == 'imagenet':
            return get_torchvision_transforms('imagenet', input_type)
        if model_type == 'vissl':
            return get_vissl_transforms(input_type)
        if model_type == 'taskonomy':
            return get_taskonomy_transforms(input_type)

    if model_type not in cached_model_types:
        if model_source == 'torchvision':
            return transform_options[model_source](model_type, input_type)
        if model_source in ['timm', 'clip', 'detectron']:
            return transform_options[model_source](model_name, input_type)
        if model_source in ['taskonomy', 'vissl', 'dino', 'midas']:
            return transform_options[model_source](input_type)

    if model_query not in list(get_model_options()) + list(model_types):
         raise ValueError('No reference available for this model query.')


def check_model(model_string, model = None):
    if not isinstance(model_string, str):
        model = model_string
    model_options = get_model_options()
    if model_string not in model_options and model == None:
        raise ValueError('model_string not available in prepped models. Please supply model object.')