from flashtorch.utils import apply_transforms, load_image
from flashtorch.utils import (denormalize,
                              format_for_plotting,
                              standardize_and_clip)
from flashtorch.saliency import Backprop
from flashtorch.activmax import GradientAscent
import copy
import torch.nn as nn


class NNManager:
    def __init__(self, NN, EPOCHS=3):
        # variables
        self.model = NN

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def image_to_tensor(self, input_):
        return apply_transforms(input_)

    def saliency_map(self, input_, target_class, guided=False, use_gpu=False, figsize=(16, 4), cmap='viridis', alpha=.5,
                     return_output=False, imageReady=False):
        temp_model = copy.deepcopy(
            self.model)  # we make a copy so the forward- and backward hooks won't be added to the original model.
        backprop = Backprop(temp_model)
        if not imageReady:
            input_ = apply_transforms(input_)
        backprop.visualize(input_, target_class, guided, use_gpu)

        """
    In this method, I will use the flashtorch code for activation_maximisation, but I will try to merge it into 1 method.
    Input variables:
        int           img_size      = image size.
        float         lr            = learning rate.
        bool          use_gpu       = use gpu.
        int           conv_layer    = The integer that points to the convolutional layer were activation maximisation will be executed on.
        bool          random        = random select a filter of a specific layer.
        bool          return_output = return the output if you want to grasp the optimized data.
        array/int     filter       = the requested filter(s)
    """

    def activation_maximisation(self, img_size=224, lr=1., use_gpu=False, filters=None, conv_layer_int=None,
                                random=False, return_output=False):
        g_ascent = GradientAscent(self.model.features, img_size=img_size, lr=lr, use_gpu=use_gpu)

        if conv_layer_int is not None:
            conv_layer = self.model.features[conv_layer_int]
            if ((filters is not None) and (not random)):
                return_value = g_ascent.visualize(conv_layer, filters,
                                                  title=('one convolutional layer is shown, filters are chosen: '),
                                                  return_output=True)
            else:
                return_value = g_ascent.visualize(conv_layer,
                                                  title=('one convolutional layer is shown, filters are at random: '),
                                                  return_output=True)

        else:
            if ((filters is not None) and (not random)):
                for feature in self.model.features:
                    if isinstance(feature, nn.modules.conv.Conv2d):
                        return_value = g_ascent.visualize(feature, filters, title=(
                            'All convolutional layers are shown, filters are chosen: '), return_output=True)

            else:
                for feature in self.model.features:
                    if isinstance(feature, nn.modules.conv.Conv2d):
                        return_value = g_ascent.visualize(feature, title=(
                            'All convolutional layers are shown, filters are at random: '), return_output=True)

        if return_output:
            return return_value

    def deepdream(self, img_path, conv_layer_int, filter_idx, img_size=224, lr=.1, num_iter=20, figsize=(4, 4),
                  title='DeepDream', return_output=False, use_gpu=False):
        g_ascent = GradientAscent(self.model.features, img_size=img_size, lr=lr, use_gpu=use_gpu)
        layer = self.model.features[conv_layer_int]
        return_value = g_ascent.deepdream(img_path, layer, filter_idx, lr=lr, num_iter=num_iter, figsize=figsize,
                                          title=title, return_output=True)
        if return_output:
            return return_value
