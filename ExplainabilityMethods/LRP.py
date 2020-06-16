import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import ModelFunctions as mf
import math
from PIL import Image
from ExplainabilityMethods.flashtorch.utils import (apply_transforms,
                                                    denormalize,
                                                    format_for_plotting,
                                                    standardize_and_clip)
import numpy
import operator


class LRP:
    """
    This class will provide an interface to perform LRP for both CNN models as LRP models.
    There are certain restrictions to the models, like that CNN models may not posses Adaptive pool layers
    and MLP models need to have an integer value as a root square from the input layer size. With other words
    the input layer for the MLP model must coincide with that of a square image.

    :param model                    A pytroch MLP or CNN model
    :param input_activation_values  All the activa
    :param rho
    :param current_layer
    :param relevance
    """
    def __init__(self):
        self.model = None
        self.input_activation_values = []
        self.rho = "lin"
        self.current_layer = None
        self.relevance = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def lrp(self, model, _input, debug=False, _return=False, rho="lin", model_type="MLP", size=224, cmap='viridis',
            alpha=.5, figsize=(16, 4)):  # todo check if it is really called lin
        """
        This method will calculate the relevances and plot these.
        :param model:       a pytorch model to perform LRP to. must be a MLP model or LRP model.
        :param _input:      (tensor or an instance of PIL.Image.Image) the input used to feed
                            the network and perform LRP from.
        :param debug:       (bool) True if you want debug statements print to the terminal, false
                            otherwise.
        :param _return:     (bool) True if you want the relevance to be returned, False otherwise.
        :param rho:         (String) 'lin' if you want to use a Linear function for the Linear
                            layers, 'relu' if you want to use a ReLU function.
        :param model_type:  (String) 'Linear' if the model contains only Linear layers and is
                            with other words an MLP network, 'Convolutional' if the network
                            Contains Convolutional layers and is a CNN model.
        :param size:        (int) In case the model is a CNN, this will rescale the input image
                            to size x size
        :param cmap:        (str, optional, default='viridis): The color map of the
                            gradients plots. See avaialable color maps `here
                            <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_.
        :param alpha:       (float, optional, default=.5): The alpha value of the max
                            gradients to be jaxaposed on top of the input image.
        :param figsize:     (tuple, optional, default=(16, 4)): The size of the plot.
        :return:            the relevances in case _return is True.
        """

        # make sure that you won't adjust the original model by registering hooks.
        self.model = copy.deepcopy(model)
        self.model.eval()

        # register activation hooks on all Linear, Convolutional and Pool layers.
        # This will make it possible to save all activation values that will be
        # used for layerwise relevance propagation.
        self.register_activation_hook()
        lin_layers = mf.get_all_lin_layers(self.model)
        if model_type == "Convolutional":
            conv_layers = mf.get_all_conv_layers(self.model)
            pool_layers = mf.get_all_pool_layers(self.model)

        # set the rho function to the correct function and set current_layer to 0.
        self.rho = rho
        self.current_layer = 0

        # make sure that the input is of a correct type, if not, change this input.
        if model_type == "MLP":
            if isinstance(_input, Image.Image) or len(list(_input.view(-1))) != lin_layers[0][1].in_features:
                _input = mf.apply_transforms(_input, size=int(math.sqrt(lin_layers[0][1].in_features)))
            else:
                _input = _input.view(-1, lin_layers[0][1].in_features)
        elif model_type == "Convolutional":
            _input = apply_transforms(_input, size)
        else:
            print("Not a valid model_type.")

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Input")
            print("-------------------------------------------------------------------------------------------------\n")
            print(_input.dtype)
            print(_input.size())

        # make sure there are no activation values at before starting the forward propagation
        self.input_activation_values.clear()
        _input.requires_grad_(True)

        # perform a forward propagation from the input data. The activation values will be saved in a list.
        output = self.model(_input)

        # save all layers so it is easy to iterate over them in the reverse order for RLP.
        layers = []
        for _, module in self.model.named_modules():
            layers.append(module)

        print("\n-------------------------------------------------------------------------------------------------")
        print("The value that the neural network thinks that corresponds to this input with its belief")
        print("-------------------------------------------------------------------------------------------------\n")
        print(output.topk(1, dim=1))
        if debug:
            print("The total output is:")
            print(output)
            print("With shape:")
            print(output.size())

            print("\n-------------------------------------------------------------------------------------------------")
            print("All layers of this module")
            print("-------------------------------------------------------------------------------------------------\n")
            for layer in layers:
                print(layer)

            print("\n-------------------------------------------------------------------------------------------------")
            print("The lin layers of this module")
            print("-------------------------------------------------------------------------------------------------\n")
            for layer in lin_layers:
                print(layer)

            if model_type == "Convolutional":
                print(
                    "\n-------------------------------------------------------------------------------------------------")
                print("The conv layers of this module")
                print(
                    "-------------------------------------------------------------------------------------------------\n")
                for layer in conv_layers:
                    print(layer)
                print(
                    "\n-------------------------------------------------------------------------------------------------")
                print("The pool layers of this module")
                print(
                    "-------------------------------------------------------------------------------------------------\n")
                for layer in pool_layers:
                    print(layer)

            print("\n-------------------------------------------------------------------------------------------------")
            print("The input values of each layer (output activations of previous layer)")
            print("-------------------------------------------------------------------------------------------------\n")
            i = 1
            for layer in self.input_activation_values:
                print("Activation values of layer %d" % i)
                print(layer.size())
                # print(layer)
                print()  # empty line
                i += 1

            print("\n-------------------------------------------------------------------------------------------------")
            print("Preparation for backpropagation")
            print("-------------------------------------------------------------------------------------------------\n")
            print("The current layer is: %d\n" % self.current_layer)

        # set the inital value of the relevance tensor, this means that we need the activation value
        # of one neuron of the last module. We will take the neuron with the highest activation. i.e.
        # the category the neural networks beliefs is the correct one for the specific given input
        # image
        self.relevance = torch.zeros(output.size())
        maximum = output.topk(1, dim=1)
        self.relevance[0][maximum[1].item()] = maximum[0].item()

        if debug:
            print("The relevance before backprop is:")
            print(self.relevance)
            print("With first tensor size:")
            print(self.relevance.size())
            print("The command Output.shape[-1] gives:")
            print(output.shape[-1])
            print("The maximum is:")
            print(maximum)
            print(output[0][maximum[1]])

            print("\n-------------------------------------------------------------------------------------------------")
            print("Backprop")
            print("-------------------------------------------------------------------------------------------------\n")

        # perform layerwise relevance propagation.
        for module in reversed(layers):
            self.relevance_layer_calculation(module)

        if debug:
            print("\nRelevance content after backprop and sizes of all layers: \n")
            print(self.relevance)
            print(self.relevance.size())

            print("\nCheck the conservation of relevance:")

            print("\nsum of all relevances: %f" % self.relevance.sum().item())
            print(int(math.sqrt(len(list(self.relevance[0])))))

        #plot the relevances.
        if model_type == "MLP":
            subplots = [
                # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
                ('Input image',
                 [(_input.view(int(math.sqrt(len(list(self.relevance[0])))),
                                           int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy(), None, None)]),
                ('Relevance',
                 [(self.relevance.view(int(math.sqrt(len(list(self.relevance[0])))),
                                           int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy(),
                   None,
                   None)]),
                ('Overlay',
                 [(_input.view(int(math.sqrt(len(list(self.relevance[0])))),
                                           int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy(), None, None),
                  (self.relevance.view(int(math.sqrt(len(list(self.relevance[0])))),
                                           int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy(),
                   cmap,
                   alpha)])
            ]
        else:
            subplots = [
                # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
                ('Input image',
                 [(format_for_plotting(denormalize(_input)), None, None)]),
                ('Relevance',
                 [(format_for_plotting(standardize_and_clip(self.relevance)),
                   None,
                   None)]),
                ('Overlay',
                 [(format_for_plotting(denormalize(_input)), None, None),
                  (format_for_plotting(standardize_and_clip(self.relevance)),
                   cmap,
                   alpha)])
            ]

        fig = plt.figure(figsize=figsize)

        for i, (title, images) in enumerate(subplots):
            ax = fig.add_subplot(1, len(subplots), i + 1)
            ax.set_axis_off()
            ax.set_title(title)

            for image, cmap, alpha in images:
                ax.imshow(image, cmap=cmap, alpha=alpha)
        plt.show()

        if _return:
            return self.relevance

    def function(self, _input):
        """
        This is the rho function that will be performed for all Linear layers.
        :param _input: The tensor to perform a funtion to.
        :return: A tensor after the function is applied
        """
        return_value = None
        if self.rho is "lin" or self.rho is 'lin':
            return_value = _input  # standard asume lin to be used
        elif self.rho is "relu" or self.rho is 'relu':
            return_value = nn.functional.relu(_input)
        return return_value

    def relevance_layer_calculation(self, module):
        """
        This method will propagate the relevance through one layer.

        You can find more information about layerwise relevance propagation at
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable

        I used the site below to get inspiration how to implement LRP.
        http://www.heatmapping.org/tutorial/

        :param module: The module where to propagate the relevance through.
        """
        if isinstance(module, nn.Linear):
            # creating a temporary new relevance of the correct size.
            layer_relevance = torch.zeros(1, module.in_features)

            # when this is the last layer to perform LRP from (the first layer of the model),
            # We need to execute another function. All functions (for both the non-last as
            # last layers) can be found in the second link above.
            if self.current_layer > 1:
                # Step 1
                z = torch.add(
                    torch.mul(self.input_activation_values[self.current_layer - 1], self.function(module.weight)).sum(
                        dim=1), sys.float_info.epsilon)  # another epsilon value  can be used as stabiliser.
                # Step 2
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    # Step 3
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()
                    # Step 4
                    layer_relevance[0][j] = torch.mul(
                        self.input_activation_values[self.current_layer - 1][0][j].item(), c)

            else:  # We will make a connection with the image.
                # Step 1
                z = (torch.mul(self.input_activation_values[self.current_layer - 1], module.weight) - torch.mul(
                    torch.clamp(module.weight, min=0),
                    0) - torch.mul(torch.clamp(module.weight, max=0),
                                   1)).sum(dim=1)
                # Step 2
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    # Step 3
                    c = torch.mul((torch.mul(self.input_activation_values[self.current_layer - 1][0][j],
                                             module.weight[:, j]) - torch.mul(
                        torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0),
                                                                                1)), s).sum()
                    # Step 4
                    layer_relevance[0][j] = c
            # Saving the temp relevance tho the class variable and setting the current_layer -1.
            self.relevance = layer_relevance
            self.current_layer -= 1
            print(self.relevance.sum())

        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d):
            module._forward_hooks.clear()  # may not be removed!!
            self.input_activation_values[self.current_layer - 1] = (
                self.input_activation_values[self.current_layer - 1].data).requires_grad_(True)

            def newlayer(layer, g):
                layer = copy.deepcopy(layer)
                try:
                    layer.weight = nn.Parameter(g(layer.weight))
                except AttributeError:
                    pass
                try:
                    layer.bias = nn.Parameter(g(layer.bias))
                except AttributeError:
                    pass
                return layer

            if self.current_layer > 1:
                def rho(p):
                    return p

                def incr(var):
                    return var  # + 1e-9

                z = incr(newlayer(module, rho).forward(self.input_activation_values[self.current_layer - 1]))  # step 1
                s = (self.relevance / z).data  # step 2
                (z * s).sum().backward()
                c = self.input_activation_values[self.current_layer - 1].grad  # step 3
                # Saving the temp relevance tho the class variable
                self.relevance = (self.input_activation_values[self.current_layer - 1] * c).data  # step 4

            else:  # We moeten nu de connectie maken met de afbeelding.
                # create an array of pixel values
                mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
                std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
                lb = (self.input_activation_values[self.current_layer - 1].data * 0 + (0 - mean) / std).requires_grad_(
                    True)
                hb = (self.input_activation_values[self.current_layer - 1].data * 0 + (1 - mean) / std).requires_grad_(
                    True)

                z = module.forward(self.input_activation_values[self.current_layer - 1]) + 1e-9  # step 1 (a)
                z -= newlayer(module, lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
                z -= newlayer(module, lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
                s = (self.relevance / z).data  # step 2
                (z * s).sum().backward()
                c, cp, cm = self.input_activation_values[self.current_layer - 1].grad, lb.grad, hb.grad  # step 3
                # Saving the temp relevance tho the class variable
                self.relevance = (
                            self.input_activation_values[self.current_layer - 1] * c + lb * cp + hb * cm).data  # step 4
            # setting the current_layer -1.
            self.current_layer -= 1
            print(self.relevance.sum())


        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AdaptiveMaxPool2d):
            module.return_indices = True
            module._forward_hooks.clear()
            output, indices = module.forward(self.input_activation_values[self.current_layer - 1])
            if isinstance(module, nn.MaxPool2d):
                stride = module.stride
                kernel_size = module.kernel_size
            else:
                input_size = tuple((self.input_activation_values[self.current_layer - 1].size()[2],
                                    self.input_activation_values[self.current_layer - 1].size()[3]))
                stride = tuple(map(operator.floordiv, input_size, module.output_size))  # input_size // output_size
                temp = tuple(map(operator.mul, tuple(var - 1 for var in module.output_size), stride))
                kernel_size = tuple(map(operator.sub, input_size, temp))
            unpool = nn.MaxUnpool2d(kernel_size, stride)
            # Saving the temp relevance tho the class variable and setting the current_layer -1.
            self.relevance = unpool(self.relevance.view(output.size()), indices,
                                    output_size=self.input_activation_values[self.current_layer - 1].size())
            self.current_layer -= 1
            print(self.relevance.sum())

    #
    # ----------------------------------------------Register Hooks------------------------------------------------------
    #
    def register_activation_hook(self):
        """
        This method will first define a function that need to be performed on every layer where this
        hook is placed on. Afterwards we will iterate over all modules (layers) to see for each layer if
        this is a layer where the hook needs to be placed on.
        """
        def activation_hook(module, input_, output):
            self.input_activation_values.append(input_[0])  # input[0] is a tuple but we want the tensor within.
            self.current_layer += 1

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or \
                    isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv3d) or \
                    isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveMaxPool2d):
                layer.register_forward_hook(activation_hook)
