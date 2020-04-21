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

"""### Saliency

Saliency map with an owl (while using the alexnet model).
This gives us two rows, the first one is without guidance and the second one with guidance (see: `Striving for Simplicity: The All Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`).
"""
if run_saliency:
    netw.saliency_map(image, 24, guided=False, use_gpu=False)
    plt.show()
    netw.saliency_map(image, 24, guided=True, use_gpu=False)
    plt.show()

"""Saliency map but with a peacock (with the alexnet model)."""
if run_saliency:
    netw.saliency_map(image2, 84, guided=False, use_gpu=False)
    plt.show()
    netw.saliency_map(image2, 84, guided=True, use_gpu=False)
    plt.show()

"""Saliency map from both an own and a peacock while using the densenet model."""
if run_saliency:
    netw2.saliency_map(image, 24, guided=False, use_gpu=False)
    plt.show()
    netw2.saliency_map(image, 24, guided=True, use_gpu=False)
    plt.show()
    netw2.saliency_map(image2, 84, guided=False, use_gpu=False)
    plt.show()
    netw2.saliency_map(image2, 84, guided=True, use_gpu=False)
    plt.show()  # make sure that everything is plotted en the next image will be plotted in a new window.

"""### Activation_maximisation

#### *all Convolutional layers*

This is the most general form for activation maximisation. Every convolutional layer will have a turn to show 4 random filters. (4 is the default number of features in flashtorch.
"""
if run_activation_max:
    netw.activation_maximisation(use_gpu=True)
    plt.show()

"""The code below will give the same result as above, but ofcoarse with some randomness in the filters."""
if run_activation_max:
    netw.activation_maximisation(use_gpu=True, random=True)
    plt.show()

"""Now we will select the filters that we want to see."""
if run_activation_max:
    netw.activation_maximisation(use_gpu=True, filters=[2, 20])
    plt.show()

"""#### *One convolutional layer*

From now on we select the convolutional layer on which we will preform activation maximisation.
"""
if run_activation_max:
    # list(model_alexnet.features.named_children()) #f.e. layer 3 is a conv2d layer
    netw.activation_maximisation(use_gpu=True, conv_layer_int=3)
    plt.show()

"""Now we will specify which filter we want to see."""
if run_activation_max:
    #list(model_vgg.features.named_children()) # f.e. layer 28 is a convolutional layer
    netw3.activation_maximisation(use_gpu=True, conv_layer_int=28, filters=5)
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
        ischanged = False
        while _input != "!exit" and not ischanged:
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
                ischanged = True
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
                print("The model is not changed..")
            else:
                print("Not a valid command/model.")
    elif _input == "!showModel":
        print(model)
    elif _input == "!quit":
        running = False
    elif modelType == "Convolutional":
        if _input == "!saliency":
            print("Saliency is running...")  # todo
        elif _input == "!activmax":
            print("Activation maximisation is running...")  # todo
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

