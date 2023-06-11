
import numpy as np
from warnings import warn
from tqdm.auto import tqdm as tqdm
from collections import defaultdict, OrderedDict
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


from packages import model_options as MO

device = torch.device('cuda:0')
def prep_model_for_extraction(model):
    if model.training:
        model = model.eval()
    if not next(model.parameters()).is_cuda:
        if torch.cuda.is_available():

            model = model.to(device)

    return(model)

def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in module for module in module_list) + 1)

    return '-'.join([class_name, class_count])

def get_inputs_sample(inputs):
    if isinstance(inputs, torch.Tensor):
        input_sample = inputs[:3]

    if isinstance(inputs, DataLoader):
        input_sample = next(iter(inputs))[:3]

    return input_sample

def get_feature_maps_(model, inputs):
    model = prep_model_for_extraction(model)

    def register_hook(module):
        def hook(module, input, output):
            module_name = get_module_name(module, feature_maps)
            feature_maps[module_name] = output

        if not isinstance(module, nn.Sequential):
            if not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))

    feature_maps = OrderedDict()
    hooks = []

    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()

    return(feature_maps)

def check_for_input_axis(feature_map, input_size):
    axis_match = [dim for dim in feature_map.shape if dim == input_size]
    return True if len(axis_match) == 1 else False
def reset_input_axis(feature_map, input_size):
    input_axis = feature_map.shape.index(input_size)
    return torch.swapaxes(feature_map, 0, input_axis)

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)
def remove_duplicate_feature_maps(feature_maps, method = 'hashkey', return_matches = False, use_tqdm = False):
    matches, layer_names = [], list(feature_maps.keys())

    if method == 'iterative':

        target_iterator = tqdm(range(len(layer_names)), leave = False) if use_tqdm else range(len(layer_names))

        for i in target_iterator:
            for j in range(i+1,len(layer_names)):
                layer1 = feature_maps[layer_names[i]].flatten()
                layer2 = feature_maps[layer_names[j]].flatten()
                if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                    if layer_names[j] not in matches:
                        matches.append(layer_names[j])

        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key not in matches}

    if method == 'hashkey':

        target_iterator = tqdm(layer_names, leave = False) if use_tqdm else layer_names
        layer_lengths = [len(tensor.flatten()) for tensor in feature_maps.values()]
        random_tensor = torch.rand(np.array(layer_lengths).max())

        tensor_dict = defaultdict(lambda:[])
        for layer_name in target_iterator:
            target_tensor = feature_maps[layer_name].flatten()
            tensor_dot = torch.dot(target_tensor, random_tensor[:len(target_tensor)])
            tensor_hash = np.array(tensor_dot).tobytes()
            tensor_dict[tensor_hash].append(layer_name)

        matches = [match for match in list(tensor_dict.values()) if len(match) > 1]
        layers_to_keep = [tensor_dict[tensor_hash][0] for tensor_hash in tensor_dict]

        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key in layers_to_keep}

    if return_matches:
        return(deduplicated_feature_maps, matches)

    if not return_matches:
        return(deduplicated_feature_maps)


def get_feature_maps(model, inputs, layers_to_retain = None, remove_duplicates = False):
    model = prep_model_for_extraction(model)
    enforce_input_shape = False

    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if layers_to_retain is None or module_name in layers_to_retain:
                    if isinstance(output, torch.Tensor):
                        #outputs = output.cpu().detach().type(torch.FloatTensor)
                        outputs = output
                        if enforce_input_shape:
                            if outputs.shape[0] == inputs.shape[0]:
                                feature_maps[module_name] = outputs



                            if outputs.shape[0] != inputs.shape[0]:
                                if check_for_input_axis(outputs, inputs.shape[0]):
                                    outputs = reset_input_axis(outputs, inputs.shape[0])
                                    feature_maps[module_name] = outputs
                                if not check_for_input_axis(outputs, inputs.shape[0]):
                                    feature_maps[module_name] = None
                                    warn('Ambiguous input axis in {}. Skipping...'.format(module_name))
                        if not enforce_input_shape:
                            feature_maps[module_name] = outputs
                if layers_to_retain is not None and module_name not in layers_to_retain:
                    feature_maps[module_name] = None

            module_name = get_module_name(module, feature_maps)

            if not any([isinstance(output, type_) for type_ in (tuple,list)]):
                process_output(output, module_name)

            if any([isinstance(output, type_) for type_ in (tuple,list)]):
                for output_i, output_ in enumerate(output):
                    module_name_ = '-'.join([module_name, str(output_i+1)])
                    process_output(output_, module_name_)

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    feature_maps = OrderedDict()
    hooks = []

    model.apply(convert_relu)
    model.apply(register_hook)
    #with torch.no_grad():
    #    model(inputs)
    model(inputs)

    for hook in hooks:
        hook.remove()

    feature_maps = {map:features for (map,features) in feature_maps.items()
                        if features is not None}

    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)

    return(feature_maps)


def get_empty_feature_maps(model, inputs = None, input_size=(3,224,224), dataset_size=3,
        layers_to_retain = None, remove_duplicates = False, names_only=False):


    if inputs is not None:
        inputs = get_inputs_sample(inputs)

    if inputs is None:
        inputs = torch.rand(3, *input_size)

    if next(model.parameters()).is_cuda:
        inputs = inputs.to(device)

    empty_feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
    for i in empty_feature_maps.keys():
        if i != 'VisionTransformer-1':
            empty_feature_maps[i] = empty_feature_maps[i].permute(1,0,2)

    for map_key in empty_feature_maps:
        empty_feature_maps[map_key] = torch.empty(dataset_size, *empty_feature_maps[map_key].shape[1:])

    if names_only == True:
        return list(empty_feature_maps.keys())

    if names_only == False:
        return empty_feature_maps


def get_prepped_model(model_string):
    model_options = MO.get_model_options()
    model_call = model_options[model_string]['call']
    model = eval(model_call)
    model = model.eval()
    model = model.to(device)


    return(model)

def get_all_feature_maps(model, inputs, layers_to_retain=None, remove_duplicates=True,
                         include_input_space = False, flatten=True, numpy=True, use_tqdm = True):

    MO.check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)

    if isinstance(inputs, DataLoader):
        input_size, dataset_size, start_index = inputs.dataset[0].shape, len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model, next(iter(inputs))[:3], input_size,
                                              dataset_size, layers_to_retain)

        if include_input_space:
            input_map = {'Input': torch.empty(dataset_size, *input_size)}
            feature_maps = {**input_map, **feature_maps}


        for imgs in tqdm(inputs, desc = 'Feature Extraction (Batch)') if use_tqdm else inputs:
            imgs = imgs.to(device) if next(model.parameters()).is_cuda else imgs
            batch_feature_maps = get_feature_maps(model, imgs, layers_to_retain )
            for i in batch_feature_maps.keys():
                if i != 'VisionTransformer-1':
                    batch_feature_maps[i] = batch_feature_maps[i].permute(1, 0, 2)

            if include_input_space:
                batch_feature_maps['Input'] = imgs.cpu()

            for map_i, map_key in enumerate(feature_maps):
                feature_maps[map_key][start_index:start_index+imgs.shape[0],...] = batch_feature_maps[map_key]
            start_index += imgs.shape[0]

    if not isinstance(inputs, DataLoader):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device) if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)

        if include_input_space:
            feature_maps = {**{'Input': inputs.cpu()}, **feature_maps}

    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)

    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)

    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].detach().numpy()

    return feature_maps