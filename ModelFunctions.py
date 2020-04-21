import torch.nn as nn


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



