import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


def get_model_type(model):
    class ModelTypeClass:
        model_type = "Linear"

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                        isinstance(module[1], nn.Conv3d):
                    ModelTypeClass.model_type = "Convolutional"

    for module in model.named_children():
        if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                isinstance(module[1], nn.Conv3d):
            ModelTypeClass.model_type = "Convolutional"
        iterative_layer_checking(module[1])

    return ModelTypeClass.model_type

def get_last_layer(model):
    class Layer:
        last_layer = None

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                Layer.last_layer = module[1]

    for module in model.named_children():
        Layer.last_layer = module[1]
        iterative_layer_checking(module[1])

    return Layer.last_layer

def get_last_conv_layer(model):
    class ConvLayer:
        last_layer = None

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                        isinstance(module[1], nn.Conv3d):
                    ConvLayer.last_layer = module[1]

    for module in model.named_children():
        if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                isinstance(module[1], nn.Conv3d):
            ConvLayer.last_layer = module[1]
        iterative_layer_checking(module[1])

    return ConvLayer.last_layer


def get_all_conv_layers(model):
    class ConvLayers:
        layers = []

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                        isinstance(module[1], nn.Conv3d):
                    ConvLayers.layers.append(module[1])

    for module in model.named_children():
        if isinstance(module[1], nn.Conv1d) or isinstance(module[1], nn.Conv2d) or \
                isinstance(module[1], nn.Conv3d):
            ConvLayers.layers.append(module[1])
        iterative_layer_checking(module[1])

    return ConvLayers.layers


def get_all_lin_layers(model):
    class LinLayers:
        layers = []

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                if isinstance(module[1], nn.Linear):
                    LinLayers.layers.append(module)

    for module in model.named_children():
        if isinstance(module[1], nn.Linear):
            LinLayers.layers.append(module)
        iterative_layer_checking(module[1])

    return LinLayers.layers


def get_all_pool_layers(model):
    class PoolLayers:
        layers = []

    def iterative_layer_checking(layer):
        for module in layer.named_children():
            if module is not None:
                iterative_layer_checking(module[1])
                if isinstance(module[1], nn.AvgPool1d) or isinstance(module[1], nn.AvgPool2d) or \
                        isinstance(module[1], nn.AvgPool3d) or isinstance(module[1], nn.MaxPool1d) or \
                        isinstance(module[1], nn.MaxPool2d) or isinstance(module[1], nn.MaxPool3d):
                    PoolLayers.layers.append(module[1])

    for module in model.named_children():
        if isinstance(module[1], nn.AvgPool1d) or isinstance(module[1], nn.AvgPool2d) or \
                isinstance(module[1], nn.AvgPool3d) or isinstance(module[1], nn.MaxPool1d) or\
                isinstance(module[1], nn.MaxPool2d) or isinstance(module[1], nn.MaxPool3d):
            PoolLayers.layers.append(module[1])
        iterative_layer_checking(module[1])

    return PoolLayers.layers

def apply_transforms(image, size=28):
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485]
    stds = [0.229]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = (1 - transform(image)).view(-1, size * size)  # make sure the highlighted feature is the text itself.

    tensor.requires_grad = True

    return tensor
