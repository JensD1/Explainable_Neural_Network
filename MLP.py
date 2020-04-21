import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import ModelFunctions as mf


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

    def lrp(self, _input, debug=False, _return=False, rho="lin"):
        # todo veralgemenen zodat verschillende inputs gegeven kunnen worden en verschillende netwerken
        self._input = _input
        if debug:
            print("\n-------------------------------------------------------------------------------------------------")
            print("Input")
            print("-------------------------------------------------------------------------------------------------\n")
            print(self._input.dtype)
            print(self._input.size())

        plt.imshow(self._input.view(28, 28))
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
        for name, layer in self.lin_layers:
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

        plt.imshow(self.relevance[len(self.relevance) - 1].view(28, 28))
        plt.show()

        if _return:
            return self.relevance

    def function(self, _input):
        return_value = None
        if self.rho is "lin":
            return_value = _input  #standard asume lin to be used
        elif self.rho is "relu":
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

    # todo:
    # efficienter maken door enkel tensor bewerkingen uit te voeren
    # Zorgen dat eender welk neuraal netwerk werkt!
    def register_backward_lin_hook(self):
        def backward_hook(module, grad_input, grad_output):
            # maak een lege tensor waarbij de grote overeenkomt met het aantal neuronen van de lager gelegen layer.
            layer_relevance = torch.zeros(1, module.in_features)
            if self.current_layer > 1:
                for j in range(module.in_features):
                    cj = 0
                    for k in range(module.out_features):
                        zk = (self.output_activation_values[self.current_layer - 2] * self.function(module.weight[k])).sum().item() + sys.float_info.epsilon  # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                        sk = self.relevance[len(self.relevance) - 1][0][k].item() / zk  # de relevance van de huidige k layer
                        cj += self.function(module.weight[k][j]).item() * sk
                    layer_relevance[0][j] = self.output_activation_values[self.current_layer - 2][0][j].item() * cj

            else:  # We moeten nu de connectie maken met de afbeelding.
                for j in range(module.in_features):
                    cj = 0
                    for k in range(module.out_features):
                        zk = (self._input * module.weight[k] - 0 * nn.functional.relu(module.weight[k]) + 1 * nn.functional.relu(-module.weight[k])).sum().item()  # self.currentlayer-2 is de layer die lager gelegen is dan de huidige
                        sk = self.relevance[len(self.relevance) - 1][0][k].item() / zk  # de relevance van de huidige k layer
                        cj += (self._input[0][j].item() * module.weight[k][j].item() - 0 * nn.functional.relu(module.weight[k][j]).item() + 1 * nn.functional.relu(-module.weight[k][j]).item()) * sk
                    layer_relevance[0][j] = cj
            self.relevance.append(layer_relevance)
            self.current_layer -= 1

        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                layer.register_backward_hook(backward_hook)
