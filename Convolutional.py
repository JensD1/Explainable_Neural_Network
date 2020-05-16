from ExplainabilityMethods.flashtorch.utils import apply_transforms
from ExplainabilityMethods.flashtorch.saliency import Backprop
from ExplainabilityMethods.flashtorch.activmax import GradientAscent
import copy
import torch.nn as nn
import ModelFunctions as mf
import ExplainabilityMethods.LRP as LRP

import matplotlib.pyplot as plt


class Convolutional:
    """This class will provide explainability methods from other classes.

    This class will only contain explainability methods that are compatible with CNN.
    In every method call, we will copy the model to perform an explainability method to.
    We do this to make sure that the forward and backward hooks won't be saved on the model and the original model is
    not adjusted by the explainability.

    Arg:
        model : A convolutional neural network model from `torchvision.models
                <https://pytorch.org/docs/stable/torchvision/models.html>`_,
                or a self created CNN.
    """
    def __init__(self):
        # variables
        self.model = None

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = copy.deepcopy(model)
        self.model.eval()

    def image_to_tensor(self, input_):
        return apply_transforms(input_)

    def saliency_map(self, input_, target_class, guided=False, use_gpu=False, figsize=(16, 4), cmap='viridis', alpha=.5,
                     return_output=False, imageReady=False):
        """
        This method will call the saliency method from flashtorch.
        Flashtorch can be found at https://github.com/MisaOgura/flashtorch, but is also implemented in this code
        in folder ExplainabilityMethods and then flashtorch.

        :param input_:          A torch tensor With shape :math:`(N, C, H, W)`
                                or an instance of PIL.Image.Image in RGB, but with param
                                imageReady set to False!
        :param target_class:    (int, optional, default=None): class where you want
                                to perform saliency for.
        :param guided:          (bool, optional, default=Fakse): If True, perform guided
                                backpropagation. See `Striving for Simplicity: The All
                                Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
        :param use_gpu:         (bool, optional, default=False): Use GPU if set to True and
                                `torch.cuda.is_available()`.
        :param figsize:         (tuple, optional, default=(16, 4)): The size of the plot.
        :param cmap:            (str, optional, default='viridis): The color map of the
                                gradients plots. See avaialable color maps `here
                                <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_.
        :param alpha:           (float, optional, default=.5): The alpha value of the max
                                gradients to be jaxaposed on top of the input image.
        :param return_output:   (bool, optional, default=False): Returns the
                                output(s) of optimization if set to True.
        :param imageReady:      When true, the apply_transforms is already done or the input is
                                already a tensor with the correct shape.
        :return:
        """
        temp_model = copy.deepcopy(
            self.model)  # we make a copy so the forward- and backward hooks won't be added to the original model.
        backprop = Backprop(temp_model)
        if not imageReady:
            input_ = apply_transforms(input_)
        backprop.visualize(input_, target_class, guided, use_gpu)  # todo return output implementation.

    def activation_maximisation(self, img_size=224, lr=1., use_gpu=False, filters=None, last_layer=True,
                                conv_layer_int=None, return_output=False):
        """
            In this method, I will use the flashtorch code for activation_maximisation, but I will try to merge it into 1 method.
            Input variables:
                :param int           img_size      = image size.
                :param float         lr            = learning rate.
                :param bool          use_gpu       = use gpu.
                :param bool          random        = random select a filter of a specific layer.
                :param bool          return_output = return the output if you want to grasp the optimized data.
                :param array/int     filters       = the requested filter(s)
                :param bool          last_layer    = True if only filters of the last layer is requested, False if the layer
                                                     is chosen with conv_layer_int.
                :param int           conv_layer_int= The idx of the convolutional layer where to perform activation maximastion
                                                     for in case the last layer is False. When last_layer is False and
                                                     conv_layer_int is None, all layers will be shown.
            """
        g_ascent = GradientAscent(self.model.features, img_size=img_size, lr=lr, use_gpu=use_gpu)
        return_value = None
        # when last layer is not requested, this will be specified with conv_layer_int.
        if not last_layer:
            if conv_layer_int is not None:
                conv_layer = self.model.features[conv_layer_int]
                if filters is not None:
                    return_value = g_ascent.visualize(conv_layer, filters,
                                                      title=('one convolutional layer is shown, filters are chosen: '),
                                                      return_output=True)
                else:
                    return_value = g_ascent.visualize(conv_layer,
                                                      title=('one convolutional layer is shown, filters are at random: '),
                                                      return_output=True)

            else:
                features = mf.get_all_conv_layers(self.model)
                if filters is not None:
                    for feature in features:
                        if isinstance(feature, nn.modules.conv.Conv2d):
                            return_value = g_ascent.visualize(feature, filters, title=(
                                'All convolutional layers are shown, filters are chosen: '), return_output=True)
                            plt.show()

                else:
                    for feature in features:
                        if isinstance(feature, nn.modules.conv.Conv2d):
                            return_value = g_ascent.visualize(feature, title=(
                                'All convolutional layers are shown, filters are at random: '), return_output=True)
                            plt.show()
        else:
            feature = mf.get_last_conv_layer(self.model)
            if filters is not None:
                if isinstance(feature, nn.modules.conv.Conv2d):
                    return_value = g_ascent.visualize(feature, filters, title=(
                        'last convolutional layer is shown, filters are chosen: '), return_output=True)
                    plt.show()

            else:
                if isinstance(feature, nn.modules.conv.Conv2d):
                    return_value = g_ascent.visualize(feature, title=(
                        'last convolutional layer is shown, filters are at random: '), return_output=True)
                    plt.show()

        if return_output:
            return return_value

    def deepdream(self, img_path, filter_idx, img_size=224, lr=.1, num_iter=20, figsize=(4, 4),
                  title='DeepDream', return_output=False, use_gpu=True):
        """
        Create a deepdream from the last layer.

        :param img_path:    (String) path to the image to perform a deepdream from.
        :param filter_idx:  The index of the target filter.
                            lr (float, optional, default=.1): The step size of optimization.
        :param img_size:    (tuple, optional, default=(4, 4)): The size of the plot.
                            Relevant in case 1 above.
        :param lr:          (float, optional, default=.1): The step size of optimization.
        :param num_iter:    (int, optional, default=30): The number of iteration for
                            the gradient ascent operation.
        :param figsize:     (tuple, optional, default=(4, 4)): The size of the plot.
                            Relevant in case 1 above.
        :param title:       (str, optional default='Conv2d'): The title of the plot.
        :param return_output:(bool, optional, default=False): Returns the
                            output(s) of optimization if set to True.
        :param use_gpu:     (bool, optional, default=False): Use GPU if set to True and
                            `torch.cuda.is_available()`.
        :return:            output (list of torch.Tensor): With dimentions
                            :math:`(num_iter, C, H, W)`. The size of the image is
                            determined by `img_size` attribute which defaults to 224.
        """

        layer = mf.get_last_conv_layer(self.model)
        g_ascent = GradientAscent(self.model.features, img_size=img_size, lr=lr, use_gpu=use_gpu)
        return_value = g_ascent.deepdream(img_path, layer, filter_idx, lr=lr, num_iter=num_iter, figsize=figsize,
                                          title=title, return_output=True)
        if return_output:
            return return_value

    def layerwise_relevance_propagation(self, _input, debug=False, _return=False, rho="lin", size=224):
        """
        Perform layerwise relevance propagation but specificly for a convolutional neural network.
        :param _input:      the input image, must be an instance of PIL.Image.Image
        :param debug:       (bool) True when you want to see debug prints, False otherwise.
        :param _return:     (bool) True if you want the relevance values.
        :param rho:         (String) denotes the rho function. 'lin' is a linear function 'relu' is a relu function.
                            This rho function will only be used for the Linear layers of the CNN model.
        :param size:        (int) the '_input' will be rescaled to an image with dimensions size x size.
        :return:            The relevance values.
        """
        _layerwise_relevance_propagation = LRP.LRP()
        _layerwise_relevance_propagation.lrp(self.model, _input, debug=debug, _return=_return, rho=rho, model_type="Convolutional", size=size) # todo implement _return
