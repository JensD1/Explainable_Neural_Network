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
import LRP as layerwiseRelevancePropagation
import NNManager as manager

import matplotlib.pyplot as plt

from PIL import Image


run_saliency = False
run_activation_max = False
run_lrp = True

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
if run_saliency or run_activation_max:
    model_alexnet = models.alexnet(pretrained=True)
    model_densenet = models.densenet201(pretrained=True)
    model_vgg = models.vgg16(pretrained=True)
    netw = manager.NNManager(model_alexnet)
    netw2 = manager.NNManager(model_densenet)
    netw3 = manager.NNManager(model_vgg)

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

"""Finally, we will create a deepdream."""
if run_activation_max:
    netw3.deepdream('images/great_grey_owl.jpg', 28, 24, use_gpu=True)
    # netw2.activation_maximisation(use_gpu=True) This type of network does not yet work.
    plt.show()

"""create MLP models"""
if run_lrp:
    model1 = net.NeuralNet(28 * 28, [128, 64], 10)
    model2 = netSeq.NeuralNetSeq()
    model1.load_state_dict(torch.load("mnist_model.pt"))
    model2.load_state_dict(torch.load("mnist_model_seq.pt"))
    lrp1 = layerwiseRelevancePropagation.LRP(model1)
    lrp2 = layerwiseRelevancePropagation.LRP(model2)

"""code runnen: hooks worden eerst geplaatst (op een copie zodat deze niet opgeslagen worden op het oorspronkelijk 
model) Vervolgens zullen we een forward functie uitproberen. """
if run_lrp:
    for images, labels in test_loader:
        relevance = lrp1.lrp(images[0].view(-1, 28 * 28), debug=True, _return=True, rho="lin")
        relevance2 = lrp1.lrp(images[0].view(-1, 28 * 28), debug=True, _return=True, rho="relu")
        break
