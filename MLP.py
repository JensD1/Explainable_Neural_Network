import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import ModelFunctions as mf
import math
from PIL import Image


class MLP:

    def __init__(self):
        self.model = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = copy.deepcopy(model)  # make sure that you won't adjust the original model by registering hooks.
        self.model.eval()
        self.register_activation_hook()
        self.register_backward_lin_hook()
        self.output_activation_values = []
        self.lin_layers = mf.get_all_lin_layers(self.model)
        self.current_layer = 0
        self.relevance = [None]
        self._input = None
        self.rho = "lin"

    def lrp(self, _input, debug=False, _return=False, rho="lin"):  # todo check if it is really called lin
        # todo veralgemenen zodat verschillende inputs gegeven kunnen worden en verschillende netwerken
        if isinstance(_input, Image.Image) or len(list(_input.view(-1))) != self.lin_layers[0][1].in_features:
            self._input = mf.apply_transforms(_input, size=int(math.sqrt(self.lin_layers[0][1].in_features)))
        else:
            self._input = _input.view(-1, self.lin_layers[0][1].in_features)

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Input")
            print("-------------------------------------------------------------------------------------------------\n")
            print(self._input.dtype)
            print(self._input.size())

        _input = self._input.detach()
        plt.imshow(_input.view(int(math.sqrt(self.lin_layers[0][1].in_features)), int(math.sqrt(self.lin_layers[0][1].in_features))))
        plt.show()

        self.output_activation_values.clear()
        output = self.model(self._input)

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("The value that the neural network thinks that corresponds to this input with its belief")
            print("-------------------------------------------------------------------------------------------------\n")
            print(output.topk(1, dim=1))

            print("\n-------------------------------------------------------------------------------------------------")
            print("All layers of this module")
            print("-------------------------------------------------------------------------------------------------\n")
            for layer in list(self.model._modules.items()):
                print(layer)

            print("\n-------------------------------------------------------------------------------------------------")
            print("The lin layers of this module")
            print("-------------------------------------------------------------------------------------------------\n")
            for layer in self.lin_layers:
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

        # Get the number of Linear layers
        amount_of_linear_layers = 0  # make sure that this is 0.
        for _ in self.lin_layers:
            amount_of_linear_layers += 1
        self.current_layer = amount_of_linear_layers

        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Preparation for backpropagation")
            print("-------------------------------------------------------------------------------------------------\n")
            print("The amount of Linear layers is: %d\n" % amount_of_linear_layers)
            print("The current layer is: %d\n" % self.current_layer)

        self.relevance = [torch.zeros(self.output_activation_values[len(self.output_activation_values) - 1].size())]
        maximum = self.output_activation_values[len(self.output_activation_values) - 1].topk(1, dim=1)
        self.relevance[0][0][maximum[1].item()] = maximum[0].item()

        if debug:
            print("The relevance before backprop is:")
            print(self.relevance)
            print("With first tensor size:")
            print(self.relevance[0].size())

        target = torch.FloatTensor(1, output.shape[-1]).zero_()
        target[0][maximum[1].item()] = 1

        if debug:
            print("\nTarget for backprop is:")
            print(target.size())
            print(target)
            print("\n-------------------------------------------------------------------------------------------------")
            print("Backprop")
            print("-------------------------------------------------------------------------------------------------\n")

        self.rho = rho  # make sure that the correct function for rho will be used.
        output.backward(target)

        if debug:
            print("\nRelevance content after backprop and sizes of all layers: \n")
            print(self.relevance)
            i = 0
            for item in self.relevance:
                print("relevance %d:" % i)
                i += 1
                print(item.size())

            print("\nCheck the conservation of relevance:")
            i = 0
            for relevance_layer in self.relevance:
                i += 1
                print("\nsum of all relevances from layer %d:" % i)
                print(relevance_layer.sum())
        print(int(math.sqrt(len(list(self.relevance[len(self.relevance) - 1][0])))))
        plt.imshow(self.relevance[len(self.relevance) - 1].view(int(math.sqrt(len(list(self.relevance[len(self.relevance) - 1][0])))), int(math.sqrt(len(list(self.relevance[len(self.relevance) - 1][0]))))))
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

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Softmax) or isinstance(layer, nn.LogSoftmax):
                layer.register_forward_hook(activation_hook)

    def register_backward_lin_hook(self):
        def backward_hook(module, grad_input, grad_output):
            # maak een lege tensor waarbij de grote overeenkomt met het aantal neuronen van de lager gelegen layer.
            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                for j in range(module.in_features):
                    #calculate the sum for a specific k
                    z = torch.add(torch.mul(self.output_activation_values[self.current_layer - 2], self.function(module.weight)).sum(dim=1), sys.float_info.epsilon)
                    s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                    c = torch.mul(s, self.function(module.weight[:, j])).sum()

                    layer_relevance[0][j] = torch.mul(self.output_activation_values[self.current_layer - 2][0][j].item(), c)

            else:  # We moeten nu de connectie maken met de afbeelding.
                for j in range(module.in_features):
                    # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                    z = (torch.mul(self._input, module.weight) - torch.mul(torch.clamp(module.weight, min=0), 0) - torch.mul(torch.clamp(module.weight, max=0), 1)).sum(dim=1)
                    s = torch.div(self.relevance[len(self.relevance) - 1][0], z)
                    c = torch.mul((torch.mul(self._input[0][j], module.weight[:, j]) - torch.mul(torch.clamp(module.weight[:, j], min=0), 0) - torch.mul(torch.clamp(module.weight[:, j], max=0), 1)), s).sum()
                    layer_relevance[0][j] = c
            self.relevance.append(layer_relevance)
            self.current_layer -= 1

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                layer.register_backward_hook(backward_hook)
