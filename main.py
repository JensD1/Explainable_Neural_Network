import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import NeuralNet as net
import NeuralNetSeq as netSeq
import MLP as multilayerperceptron
import Convolutional as convolutional
import ModelFunctions as mf

import matplotlib.pyplot as plt

from PIL import Image

print("Initializing...")

run_saliency = False
run_activation_max = False
run_lrp = False

"""## 2) Get images and models"""
if run_saliency or run_activation_max:
    image = Image.open('images/great_grey_owl.jpg')
    image2 = Image.open('images/peacock.jpg')

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    a.set_title('Original image: Great grey owl')
    a.axis('off')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    a.set_title('Original image: Peacock')
    a.axis('off')
    plt.show()

if run_lrp:
    trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

"""### Models

Loading some models to test our neural network manager.
"""
model_alexnet = models.alexnet(pretrained=True)
model_densenet = models.densenet201(pretrained=True)
model_vgg = models.vgg16(pretrained=True)
netw = convolutional.Convolutional()
netw.set_model(model_alexnet)
netw2 = convolutional.Convolutional()
netw2.set_model(model_densenet)
netw3 = convolutional.Convolutional()
netw3.set_model(model_vgg)

"""### Activation_maximisation

#### *all Convolutional layers*

This is the most general form for activation maximisation. Every convolutional layer will have a turn to show 4 random filters. (4 is the default number of features in flashtorch.
"""
if run_activation_max:
    netw.activation_maximisation(last_layer=True, use_gpu=True)
    plt.show()
    netw2.activation_maximisation(last_layer=True, use_gpu=True)
    plt.show()
    netw3.activation_maximisation(last_layer=True, use_gpu=True)
    plt.show()

"""create MLP models"""
model1 = net.NeuralNet(28 * 28, [128, 64], 10)
model2 = netSeq.NeuralNetSeq()
model1.load_state_dict(torch.load("mnist_model.pt"))
model2.load_state_dict(torch.load("mnist_model_seq.pt"))
lrp1 = multilayerperceptron.MLP()
lrp1.set_model(model1)
lrp2 = multilayerperceptron.MLP()
lrp2.set_model(model2)

"""code runnen: hooks worden eerst geplaatst (op een copie zodat deze niet opgeslagen worden op het oorspronkelijk 
model) Vervolgens zullen we een forward functie uitproberen. """
if run_lrp:
    for images, labels in test_loader:
        relevance = lrp1.lrp(images[0].view(-1, 28 * 28), debug=True, _return=True, rho="lin")
        relevance2 = lrp1.lrp(images[0].view(-1, 28 * 28), debug=True, _return=True, rho="relu")
        break

#
# ------------------------------------------------------Menu------------------------------------------------------------
#
running = True
model = None
modelType = None
convManager = convolutional.Convolutional()
mlpManager = multilayerperceptron.MLP()

availableModels = {
    "alexNet": model_alexnet,
    "denseNet": model_densenet,
    "vgg": model_vgg,
    "mlp": model1,
    "mlpSeq": model2
}
saliency_options = {
    "Guidance": True,
    "Expected": 0,
    "use_gpu": False,
    "return_output": False,
    "Alpha": 0.5,
    "image": Image.open("images/Uil.jpg")
}
activmax_options = {
    "use_gpu": True,
    "filters": None,
    "last_layer": True,
    "conv_layer_int": None,
    "return_output": False
}

while running:
    if model is None:
        print("You should select a model to use first, you can switch to another model later on!")
    _input = input("type a command: (type '!help' to view the available commands)\n")

    if _input == "!help":
        print("The general commands you can do are:")
        print("!listModels: \t\tprints a list of all available models.")
        print("!selectModel: \t\tselect the model to explain.")
        print("!showModel: \t\tshows the current model.")
        print("!addModel: \t\tadd a new model.")  # todo
        print("!deleteModel: \t\tdelete a model.")  # todo
        print("!quit: \t\t\tquits this program.")
        if model is not None:
            if modelType == "Convolutional":
                print("\nThe convolutional specific commands you can do are:")
                print("!saliency: \t\tuses the saliency explainability method.")  # todo
                print("!activmax: \t\tuses activation maximisation as explainability method.")  # todo
                print("!deepdream: \t\tcreate a deepdream from an image.")  # todo
            elif modelType == "Linear":
                print("\nThe multilayer perceptron specific commands you can do are:")
                print("!lrp: \t\t\tlayer wise relevance propagation explainability method.")  # todo
    elif _input == "!listModels":
        for tempModel in availableModels:
            print(tempModel)
    elif _input == "!selectModel":
        _exit = False
        while not _exit:
            print("Give the name of the model:")
            print("(type !list to see all possibilities.)")
            print("(type !exit if you don't want to change the model anymore.)")
            print("(type !current if you want to see the current model.")
            _input = input()
            if _input == "!list":
                for tempModel in availableModels:
                    print(tempModel)
                print()
            elif _input == "!current":
                print(model)
                print()
            elif _input in availableModels:
                _exit = True
                model = availableModels.get(_input, None)
                if model is None:
                    print("This is not an available model. You can print all available models with: !listModels.")
                    modelType = None
                else:
                    modelType = mf.get_model_type(model)
                    if modelType == "Convolutional":
                        convManager.set_model(model)
                    elif modelType == "Linear":
                        mlpManager.set_model(model)
            elif _input == "!exit":
                _exit = True
                print("The model is not changed..")
            else:
                print("Not a valid command/model.")
    elif _input == "!showModel":
        print(model)
    elif _input == "!quit":
        running = False
    elif modelType == "Convolutional":
        if _input == "!saliency":
            _exit = False
            print("You can adjust the settings, see !help for more information.")
            while not _exit:
                _input = input("Type !start to start the saliency\n")
                if _input == "!help":
                    print("The available commandos are:")
                    print("!start: \t\tstart saliency with the current options")
                    print("!listOptions: \t\tlists all the current option values.")
                    print("!exit: \t\t\texit saliency.")
                    print("You can adjust the next options:")
                    for option in saliency_options:
                        print("!", end="")
                        print(option)
                    print()
                elif _input == "!exit":
                    print("Saliency won't be performed...")
                    print()
                    _exit = True
                elif _input == "!listOptions":
                    for option in saliency_options:
                        print(option, end="")
                        print(":\t\t", end="")
                        print(saliency_options.get(option))
                    print()
                elif _input == "!Guidance":
                    _input = input("Set the value for guidance, must be True or False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        saliency_options["Guidance"] = True
                        print("Guidance is set to:", end="")
                        print(saliency_options.get("Guidance"))
                        print()
                    elif _input in ["False", "false", "f", "F"]:
                        saliency_options["Guidance"] = False
                        print("Guidance is set to:", end="")
                        print(saliency_options.get("Guidance"))
                        print()
                    else:
                        print("Not an available option...\n")
                elif _input == "!Expected":
                    saliency_options["Expected"] = int(
                        input("Give the expected outcome: (so we can notify you if the image is wrongly "
                              "categorized)\n"))
                    print("Expected is set to:", end="")
                    print(saliency_options.get("Expected"))
                    print()
                elif _input == "!Alpha":
                    saliency_options["Alpha"] = float(input("Give a new value for alpha:\n"))
                    print("Alpha is set to:", end="")
                    print(saliency_options.get("Alpha"))
                    print()
                elif _input == "!use_gpu":
                    _input = input("Set the value for use_gpu, must be True or False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        saliency_options["use_gpu"] = True
                        print("use_gpu is set to:", end="")
                        print(saliency_options.get("use_gpu"))
                        print()
                    elif _input in ["False", "false", "f", "F"]:
                        saliency_options["use_gpu"] = False
                        print("use_gpu is set to:", end="")
                        print(saliency_options.get("use_gpu"))
                        print()
                    else:
                        print("Not an available option...\n")
                elif _input == "!return_output":
                    _input = input("Set the value for return_output, must be True or False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        saliency_options["return_output"] = True
                        print("return_output is set to:", end="")
                        print(saliency_options.get("return_output"))
                    elif _input in ["False", "false", "f", "F"]:
                        saliency_options["return_output"] = False
                        print("return_output is set to:", end="")
                        print(saliency_options.get("return_output"))
                    else:
                        print("Not an available option...\n")
                elif _input == "!image":
                    path = input("Give the path to the image:\n")
                    saliency_options["image"] = Image.open(path)
                elif _input == "!start":
                    _exit = True
                    print("Saliency is running...")
                    convManager.saliency_map(saliency_options.get("image"),
                                             saliency_options.get("Expected"),
                                             guided=saliency_options.get("Guidance"),
                                             use_gpu=saliency_options.get("use_gpu"),
                                             return_output=saliency_options.get("return_output"),
                                             alpha=saliency_options.get("Alpha")
                                             )
                    plt.show()
                else:
                    print("Not an available command, type !help to see all available commands.")
        elif _input == "!activmax":
            print("You can change the requested options, type !help for more insight.")
            print("Type !start if you want to start activation maximisation.")
            _exit = False
            while not _exit:
                _input = input("Type !start to start activation maximisation\n")
                if _input == "!help":
                    print("The available commandos are:")
                    print("!start: \t\tstart activation maximisation with the current options")
                    print("!listOptions: \t\tlists all the current option values.")
                    print("!exit: \t\t\texit activation maximisation.")
                    print("You can adjust the next options:")
                    for option in activmax_options:
                        print("!", end="")
                        print(option)
                    print()
                elif _input == "!exit":
                    print("Activation maximisation won't be performed...")
                    _exit = True
                elif _input == "!listOptions":
                    for option in activmax_options:
                        print(option, end="")
                        print(":\t\t", end="")
                        print(activmax_options.get(option))
                    print()
                elif _input == "!use_gpu":
                    _input = input("Set the value for use_gpu, must be True or False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        activmax_options["use_gpu"] = True
                        print("use_gpu is set to:", end="")
                        print(activmax_options.get("use_gpu"))
                        print()
                    elif _input in ["False", "false", "f", "F"]:
                        activmax_options["use_gpu"] = False
                        print("use_gpu is set to:", end="")
                        print(activmax_options.get("use_gpu"))
                        print()
                    else:
                        print("Not an available option...\n")
                elif _input == "!return_output":
                    _input = input("Set the value for return_output, must be True or False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        activmax_options["return_output"] = True
                        print("return_output is set to:", end="")
                        print(activmax_options.get("return_output"))
                    elif _input in ["False", "false", "f", "F"]:
                        activmax_options["return_output"] = False
                        print("return_output is set to:", end="")
                        print(activmax_options.get("return_output"))
                    else:
                        print("Not an available option...\n")
                elif _input == "!last_layer":
                    _input = input("Set the value for last_layer, must be True of False:\n")
                    if _input in ["True", "true", "t", "T"]:
                        activmax_options["last_layer"] = True
                        print("last_layer is set to:", end="")
                        print(activmax_options.get("last_layer"))
                    elif _input in ["False", "false", "f", "F"]:
                        activmax_options["last_layer"] = False
                        print("last_layer is set to:", end="")
                        print(activmax_options.get("last_layer"))
                    else:
                        print("Not an available option...")
                elif _input == "!filters":
                    _input = input("Give the filter(s) you want to see.\n")
                    if _input == "None":
                        activmax_options["filters"] = None
                    else:
                        activmax_options["filters"] = int(_input) # todo: make sure also multiple filters is possible.
                elif _input == "!conv_layer_int":
                    _input = input("Give the number corresponding with the convolutional layer you want to see.\n")
                    if _input == "None":
                        activmax_options["conv_layer_int"] = None
                    else:
                        activmax_options["conv_layer_int"] = int(_input)
                elif _input == "!start":
                    _exit = True
                    print("Activation maximisation is running...")
                    convManager.activation_maximisation(use_gpu=activmax_options.get("use_gpu"),
                                                        filters=activmax_options.get("filters"),
                                                        last_layer=activmax_options.get("last_layer"),
                                                        conv_layer_int=activmax_options.get("conv_layer_int"),
                                                        return_output=activmax_options.get("return_output"))
                    plt.show()
                else:
                    print("Not an available command, type !help to see all available commands.")
        elif _input == "!deepdream":
            path = input("Give the path to the image:\n")
            filter_number = int(input("Give the requested filter number:\n"))
            print("Deepdream is running...")
            convManager.deepdream(path, filter_number, use_gpu=True)
            plt.show()
        else:
            print("This is not an available command!")
    elif modelType == "Linear":
        if _input == "!lrp":
            print("Layer wise relevance propagation is running...")  # todo
        else:
            print("This is not an available command!")
    else:
        print("This is not an available command!")
    print()

