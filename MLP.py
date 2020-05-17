import copy
from ExplainabilityMethods.flashtorch.saliency import Backprop
import ModelFunctions as mf
import math
from PIL import Image

import ExplainabilityMethods.LRP as LRP
class MLP:
    """This class will provide explainability methods from other classes.

    This class will only contain explainability methods that are compatible with MLP networks.
    In every method call, we will copy the model to perform an explainability method to.
    We do this to make sure that the forward and backward hooks won't be saved on the model and the original model is
    not adjusted by the explainability.

    Arg:
        model : A multilayer perceptron model.
    """
    def __init__(self):
        self.model = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def get_model(self):
        return self.model

    def set_model(self, model):
        """
        This method will set the class model variable.
        :param model: Must be a MLP model with only ReLU and Linear layers and layers like SoftMax.
                      The input layer of this model must be one compatible with a square image f.e. 100 x 100.
                      If the input layer has a number of neurons that has not an integer as root, LRP will not work!
        :return:
        """
        self.model = model  # make sure that you won't adjust the original model by registering hooks.

    def layerwise_relevance_propagation(self, _input, debug=False, _return=False, rho="lin"):
        """
        Perform a layerwise relevance propagation on an _input image with the model that needed to be set previously
        with the set_model method.
        :param _input: an input tensor of a in instance of PIL.Image.Image. This will be rescaled automatically.
        :param debug:   (bool) True if you want debug statements printed in the terminal.
        :param _return: (bool) True if the relevance values need to be returned.
        :param rho:     (String) 'lin' if you want a linear rho function, 'relu' if you want to use a relu function.
                        This function will be used for all Linear layers.
        :return:        The calculated relevances for a specific input image.
        """
        # the copy is already done in LRP so it is not necessary here.
        _layerwise_relevance_propagation = LRP.LRP()
        _layerwise_relevance_propagation.lrp(self.model, _input, debug=debug, _return=_return, rho=rho, model_type="MLP")

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
            lin_layers = mf.get_all_lin_layers(self.model)
            if isinstance(input_, Image.Image) or len(list(input_.view(-1))) != lin_layers[0][1].in_features:
                input_ = mf.apply_transforms(input_, size=int(math.sqrt(lin_layers[0][1].in_features)))
            else:
                input_ = input_.view(-1, lin_layers[0][1].in_features)
        backprop.visualize(input_, target_class, guided, use_gpu, model_type="MLP")  # todo return output implementation.
