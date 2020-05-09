import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import ModelFunctions as mf
import math
from PIL import Image
from ExplainabilityMethods.flashtorch.utils import apply_transforms, format_for_plotting
import numpy

class LRP:

    def __init__(self):
        self.model = None
        self.input_activation_values = []
        self.rho = "lin"
        self.current_layer = None
        self.relevance = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def lrp(self, model, _input, debug=False, _return=False, rho="lin", model_type="MLP"):  # todo check if it is really called lin
        self.model = copy.deepcopy(model)  # make sure that you won't adjust the original model by registering hooks.
        self.model.eval()
        self.register_activation_hook()
        last_layer = mf.get_last_layer(self.model)
        lin_layers = mf.get_all_lin_layers(self.model)
        if model_type == "Convolutional":
            conv_layers = mf.get_all_conv_layers(self.model)
            pool_layers = mf.get_all_pool_layers(self.model)
        self.rho = rho
        self.current_layer = 0

        # todo apply transfors van flashtorch indien convolutional
        if model_type == "MLP":
            if isinstance(_input, Image.Image) or len(list(_input.view(-1))) != lin_layers[0][1].in_features:
                _input = mf.apply_transforms(_input, size=int(math.sqrt(lin_layers[0][1].in_features)))
            else:
                _input = _input.view(-1, lin_layers[0][1].in_features)
        elif model_type == "Convolutional":
            _input = apply_transforms(_input)
        else:
            print("Not a valid model_type.")

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Input")
            print("-------------------------------------------------------------------------------------------------\n")
            print(_input.dtype)
            print(_input.size())

        _input = _input.detach()
        if model_type == "MLP":
            plt.imshow(_input.view(int(math.sqrt(lin_layers[0][1].in_features)), int(math.sqrt(lin_layers[0][1].in_features))))
        elif model_type == "Convolutional":
            plt.imshow(format_for_plotting(_input))
        plt.show()

        self.input_activation_values.clear()
        _input.requires_grad_(True)
        output = self.model(_input)
        layers = []
        for _, module in self.model.named_modules():
            layers.append(module)
        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("The value that the neural network thinks that corresponds to this input with its belief")
            print("-------------------------------------------------------------------------------------------------\n")
            print(output.topk(1, dim=1))
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
                print("\n-------------------------------------------------------------------------------------------------")
                print("The conv layers of this module")
                print("-------------------------------------------------------------------------------------------------\n")
                for layer in conv_layers:
                    print(layer)
                print("\n-------------------------------------------------------------------------------------------------")
                print("The pool layers of this module")
                print("-------------------------------------------------------------------------------------------------\n")
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

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Preparation for backpropagation")
            print("-------------------------------------------------------------------------------------------------\n")
            print("The current layer is: %d\n" % self.current_layer)

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

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Backprop")
            print("-------------------------------------------------------------------------------------------------\n")
        for module in reversed(layers):
            self.relevance_layer_calculation(module)

        if debug:
            print("\nRelevance content after backprop and sizes of all layers: \n")
            print(self.relevance)
            print(self.relevance.size())

            print("\nCheck the conservation of relevance:")

            print("\nsum of all relevances: %f" % self.relevance.sum().item())
            print(int(math.sqrt(len(list(self.relevance[0])))))
        if model_type == "MLP":
            plt.imshow(self.relevance.view(int(math.sqrt(len(list(self.relevance[0])))), int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy())
        else:
            def heatmap(R, sx, sy):

                b = 10 * ((numpy.abs(R) ** 3.0).mean() ** (1.0 / 3))

                from matplotlib.colors import ListedColormap
                my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
                my_cmap[:, 0:3] *= 0.85
                my_cmap = ListedColormap(my_cmap)
                plt.figure(figsize=(sx, sy))
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
                plt.show()
            heatmap(numpy.array(self.relevance[0]).sum(axis=0), 3.5, 3.5)
        plt.show()

        if _return:
            return self.relevance

    def function(self, _input):
        return_value = None
        if self.rho is "lin" or self.rho is 'lin':
            return_value = _input  #standard asume lin to be used
        elif self.rho is "relu" or self.rho is 'relu':
            return_value = nn.functional.relu(_input)
        return return_value

    #
    # ----------------------------------------------Register Hooks------------------------------------------------------
    #
    def register_activation_hook(self):
        def activation_hook(module, input_, output):
            self.input_activation_values.append(input_[0])  # input[0] is a tuple but we want the tensor within.
            self.current_layer += 1

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or \
                    isinstance(layer, nn.Conv3d) or isinstance(layer, nn.AvgPool1d) or \
                    isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.AvgPool3d) or \
                    isinstance(layer, nn.MaxPool1d) or isinstance(layer, nn.MaxPool2d) or \
                    isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.AdaptiveAvgPool2d) or \
                    isinstance(layer, nn.AdaptiveAvgPool1d) or isinstance(layer, nn.AdaptiveAvgPool3d):
                layer.register_forward_hook(activation_hook)

    def relevance_layer_calculation(self, module):
        if isinstance(module, nn.Linear):
            print(self.current_layer)
            print("module is Linear!")
            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                # calculate the sum for a specific k
                z = torch.add(
                    torch.mul(self.input_activation_values[self.current_layer - 1], self.function(module.weight)).sum(
                        dim=1), sys.float_info.epsilon)
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()
                    layer_relevance[0][j] = torch.mul(
                        self.input_activation_values[self.current_layer - 1][0][j].item(), c)

            else:  # We moeten nu de connectie maken met de afbeelding.
                z = (torch.mul(self.input_activation_values[self.current_layer - 1], module.weight) - torch.mul(torch.clamp(module.weight, min=0),
                                                                       0) - torch.mul(torch.clamp(module.weight, max=0),
                                                                                      1)).sum(dim=1)
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                    c = torch.mul((torch.mul(self.input_activation_values[self.current_layer - 1][0][j], module.weight[:, j]) - torch.mul(
                        torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0),
                                                                                1)), s).sum()
                    layer_relevance[0][j] = c
            self.relevance = layer_relevance
            self.current_layer -= 1
            print(self.relevance.sum().item())

        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d):
            print(self.current_layer)
            print("module is Conv!")
            module._forward_hooks.clear()  # may not be removed!!
            self.input_activation_values[self.current_layer - 1] = (self.input_activation_values[self.current_layer - 1].data).requires_grad_(True)
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

                def incr(z):
                    return z + 1e-9
                z = incr(newlayer(module, rho).forward(self.input_activation_values[self.current_layer - 1]))  # step 1
                s = (self.relevance / z).data  # step 2
                (z * s).sum().backward()
                c = self.input_activation_values[self.current_layer - 1].grad  # step 3
                self.relevance = (self.input_activation_values[self.current_layer - 1] * c).data  # step 4

            else:  # We moeten nu de connectie maken met de afbeelding.
                #create an array of pixel values
                mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
                std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
                lb = (self.input_activation_values[self.current_layer - 1].data * 0 + (0 - mean) / std).requires_grad_(True)
                hb = (self.input_activation_values[self.current_layer - 1].data * 0 + (1 - mean) / std).requires_grad_(True)

                z = module.forward(self.input_activation_values[self.current_layer - 1]) + 1e-9  # step 1 (a)
                z -= newlayer(module, lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
                z -= newlayer(module, lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
                s = (self.relevance / z).data  # step 2
                (z * s).sum().backward();
                c, cp, cm = self.input_activation_values[self.current_layer - 1].grad, lb.grad, hb.grad  # step 3
                self.relevance = (self.input_activation_values[self.current_layer - 1] * c + lb * cp + hb * cm).data  # step 4
            self.current_layer -= 1
            print(self.relevance.sum().item())

        elif isinstance(module, nn.AvgPool1d) or isinstance(module, nn.AvgPool2d) or \
                    isinstance(module, nn.AvgPool3d) or isinstance(module, nn.MaxPool1d) or \
                    isinstance(module, nn.MaxPool2d) or isinstance(module, nn.MaxPool3d) or \
                    isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.AdaptiveAvgPool1d) or \
                    isinstance(module, nn.AdaptiveAvgPool3d):
            print(self.current_layer)
            print("module is Pool!")
            module.return_indices = True
            module._forward_hooks.clear()
            output, indices = module.forward(self.input_activation_values[self.current_layer - 1])
            unpool = nn.MaxUnpool2d(module.kernel_size, module.stride)
            self.relevance = unpool(self.relevance.view(output.size()), indices)
            self.current_layer -= 1
            print(self.relevance.sum().item())

