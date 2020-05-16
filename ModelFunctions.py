import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


def get_model_type(model):
    """
    This function looks at all layers and seeks for nn.Conv1d, nn.Conv2d and nn.Conv3d modules. In case there is a
    module of this instance, than we have a CNN, otherwise a MLP network. This method is not created to find other model
    types like f.e. RNN.
    :param model: A MLP model or a CNN model.
    :return:    "Linear" if the model has no Conv layers but only Linear (ReLU, SoftMax, ...),
                otherwise returns "Convolutional"
    """
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
    """
    Get the last layer of a model
    :param model:  any type of pytorch model.
    :return: The last layer of this model.
    """
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
    """
    Get the last convolutional layer of a model.
    :param model: any kind of pytorch model.
    :return: The last convolutional layer.
    """
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
    """
    This method extracts all convolutional layers from a CNN.
    :param model: A pytorch CNN model.
    :return: A list of all convolutional layers of the model.
    """
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
    """
    This method extracts all Linear layers from a MLP network.
    :param model: A pytorch MLP network.
    :return: A list of all linear layers of the model.
    """
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
    """
        This method extracts all pool layers from a CNN. (at the moment only Avg- and MaxPool are
        implemented, AdaptingAvg- and AdaptiveMaxPool layers are not).
        :param model: A pytorch CNN model.
        :return: A list of all pool layers of the model.
        """
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
    """

    :param image: image (PIL.Image.Image or numpy array) with shape :math:`(1, H, W)` in case of numpy array
    :param size: the size of the rescaled image (size x size).
    :return: a tensor of size [1, size x size]
    """
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
