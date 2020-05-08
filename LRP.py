import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import ModelFunctions as mf
import math
import flashtorch
from PIL import Image
from flashtorch.utils import apply_transforms, format_for_plotting

class LRP:

    def __init__(self):
        self.model = None
        self.output_activation_values = []
        self.rho = "lin"
        self._input = None
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
        # we need to make sure that all activations are registered, the last layer needs to be registered too, but isn't always a softmax or something alike.
        if not isinstance(last_layer, nn.Softmax) and not isinstance(last_layer, nn.LogSoftmax):  # if there is one of these layers in the end, the current_layer will be wrong.
            self.register_activation_hook_last_layer(last_layer)
        lin_layers = mf.get_all_lin_layers(self.model)
        if model_type == "Convolutional":
            conv_layers = mf.get_all_conv_layers(self.model)
            pool_layers = mf.get_all_pool_layers(self.model)
        self.rho = rho
        self.current_layer = 0

        # todo apply transfors van flashtorch indien convolutional
        if model_type == "MLP":
            if isinstance(_input, Image.Image) or len(list(_input.view(-1))) != lin_layers[0][1].in_features:
                self._input = mf.apply_transforms(_input, size=int(math.sqrt(lin_layers[0][1].in_features)))
            else:
                self._input = _input.view(-1, lin_layers[0][1].in_features)
        elif model_type == "Convolutional":
            self._input = apply_transforms(_input)
        else:
            print("Not a valid model_type.")

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Input")
            print("-------------------------------------------------------------------------------------------------\n")
            print(self._input.dtype)
            print(self._input.size())

        _input = self._input.detach()
        if model_type == "MLP":
            plt.imshow(_input.view(int(math.sqrt(lin_layers[0][1].in_features)), int(math.sqrt(lin_layers[0][1].in_features))))
        elif model_type == "Convolutional":
            plt.imshow(format_for_plotting(_input))
        plt.show()

        self.output_activation_values.clear()
        output = self.model(self._input)
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
            print("The activation values of each layer")
            print("-------------------------------------------------------------------------------------------------\n")
            i = 1
            for layer in self.output_activation_values:
                print("Activation values of layer %d" % i)
                print(layer.size())
                print(layer)
                print()  # empty line
                i += 1

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Preparation for backpropagation")
            print("-------------------------------------------------------------------------------------------------\n")
            print("The current layer is: %d\n" % self.current_layer)

        self.relevance = torch.zeros(self.output_activation_values[len(self.output_activation_values) - 1].size())
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
        plt.imshow(self.relevance.view(int(math.sqrt(len(list(self.relevance[0])))), int(math.sqrt(len(list(self.relevance[0]))))).detach().numpy())
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
            self.output_activation_values.append(output)
            self.current_layer += 1

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Softmax) or isinstance(layer, nn.LogSoftmax) or \
                    isinstance(layer, nn.AvgPool1d) or isinstance(layer, nn.AvgPool2d) or \
                    isinstance(layer, nn.AvgPool3d) or isinstance(layer, nn.MaxPool1d) or \
                    isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.MaxPool3d) or \
                    isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer, nn.AdaptiveAvgPool1d) or \
                    isinstance(layer, nn.AdaptiveAvgPool3d):
                layer.register_forward_hook(activation_hook)

    def register_activation_hook_last_layer(self, layer):
        def activation_hook(module, input_, output):
            self.output_activation_values.append(output)
            self.current_layer += 1
        layer.register_forward_hook(activation_hook)

    def relevance_layer_calculation(self, module):
        if isinstance(module, nn.Linear):
            print("module is linear!")
            # maak een lege tensor waarbij de grote overeenkomt met het aantal neuronen van de lager gelegen layer.
            print(self.current_layer)
            print(module)
            print(self.output_activation_values[self.current_layer - 2].size())
            print(self.function(module.weight).size())

            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                # calculate the sum for a specific k
                z = torch.add(
                    torch.mul(self.output_activation_values[self.current_layer - 2], self.function(module.weight)).sum(
                        dim=1), sys.float_info.epsilon)
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()
                    layer_relevance[0][j] = torch.mul(
                        self.output_activation_values[self.current_layer - 2][0][j].item(), c)

            else:  # We moeten nu de connectie maken met de afbeelding.
                z = (torch.mul(self._input, module.weight) - torch.mul(torch.clamp(module.weight, min=0),
                                                                       0) - torch.mul(torch.clamp(module.weight, max=0),
                                                                                      1)).sum(dim=1)
                s = torch.div(self.relevance[0], z)
                for j in range(module.in_features):
                    # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                    c = torch.mul((torch.mul(self._input[0][j], module.weight[:, j]) - torch.mul(
                        torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0),
                                                                                1)), s).sum()
                    layer_relevance[0][j] = c
            self.relevance = layer_relevance
            self.current_layer -= 1

        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv3d):
            # maak een lege tensor waarbij de grote overeenkomt met het aantal neuronen van de lager gelegen layer.
            print(self.current_layer)
            print(module)
            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                # calculate the sum for a specific k
                z = torch.add(
                    torch.mul(self.output_activation_values[self.current_layer - 2], self.function(module.weight)).sum(
                        dim=1), sys.float_info.epsilon)
                s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                for j in range(module.in_features):
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()

                    layer_relevance[0][j] = torch.mul(
                        self.output_activation_values[self.current_layer - 2][0][j].item(), c)

            else:  # We moeten nu de connectie maken met de afbeelding.
                # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                z = (torch.mul(self._input, module.weight) - torch.mul(torch.clamp(module.weight, min=0),
                                                                       0) - torch.mul(torch.clamp(module.weight, max=0),
                                                                                      1)).sum(dim=1)
                s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                for j in range(module.in_features):
                    c = torch.mul((torch.mul(self._input[0][j], module.weight[:, j]) - torch.mul(
                        torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0),
                                                                                1)), s).sum()
                    layer_relevance[0][j] = c
            self.relevance = layer_relevance
            self.current_layer -= 1

        elif isinstance(module, nn.AvgPool1d) or isinstance(module, nn.AvgPool2d) or \
                    isinstance(module, nn.AvgPool3d) or isinstance(module, nn.MaxPool1d) or \
                    isinstance(module, nn.MaxPool2d) or isinstance(module, nn.MaxPool3d) or \
                    isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.AdaptiveAvgPool1d) or \
                    isinstance(module, nn.AdaptiveAvgPool3d):
            # maak een lege tensor waarbij de grote overeenkomt met het aantal neuronen van de lager gelegen layer.

            print(self.current_layer)
            print(module)

            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                # calculate the sum for a specific k
                z = torch.add(
                    torch.mul(self.output_activation_values[self.current_layer - 2], self.function(module.weight)).sum(
                        dim=1), sys.float_info.epsilon)
                s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                for j in range(module.in_features):
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()

                    layer_relevance[0][j] = torch.mul(
                        self.output_activation_values[self.current_layer - 2][0][j].item(), c)

            else:  # We moeten nu de connectie maken met de afbeelding.
                # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                z = (torch.mul(self._input, module.weight) - torch.mul(torch.clamp(module.weight, min=0),
                                                                       0) - torch.mul(torch.clamp(module.weight, max=0),
                                                                                      1)).sum(dim=1)
                s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                for j in range(module.in_features):
                    c = torch.mul((torch.mul(self._input[0][j], module.weight[:, j]) - torch.mul(
                        torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0),
                                                                                1)), s).sum()
                    layer_relevance[0][j] = c
            self.relevance = layer_relevance
            self.current_layer -= 1
