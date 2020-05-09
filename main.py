import torch
import logging

import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import MLP as multilayerperceptron
import Convolutional as convolutional
import ModelFunctions as mf

import matplotlib.pyplot as plt

from PIL import Image

print("Initializing...")

trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

#
# -------------------------------------------------Menu-Options---------------------------------------------------------
#
running = True
model = None
modelType = None
convManager = convolutional.Convolutional()
mlpManager = multilayerperceptron.MLP()

availableModels = [  # when changing something here the loadModel function should be adapted as well.
    "alexNet",
    "denseNet",
    "vgg"#,
    # "mlp",
    # "mlpSeq",
    # "mlp29"
]


def load_model(_input):
    global model
    model = None
    if _input in availableModels:
        if _input == "alexNet":
            model = models.alexnet(pretrained=True)
        elif _input == "denseNet":
            model = models.densenet201(pretrained=True)
        elif _input == "vgg":
            model = models.vgg16(pretrained=True)
        # elif _input == "mlp":
        #     loaded_model = net.NeuralNet(28 * 28, [128, 64], 10)
        #     loaded_model.load_state_dict(torch.load("mnist_model.pt"))
        # elif _input == "mlpSeq":
        #     loaded_model = netSeq.NeuralNetSeq()
        #     loaded_model.load_state_dict(torch.load("mnist_model_seq.pt"))
        # elif _input == "mlp29":
        #     loaded_model = net.NeuralNet(29 * 29, [128, 64], 10)
        #     loaded_model.load_state_dict(torch.load("mnist_model_29x29.pt"))
        else:
            try:
                exec('model = %s()' % _input, globals())
                state_dict_file_name = input("Give the state dict file name. (None if there is none).\n")
                if state_dict_file_name not in ["None", "none", "no", "No", "n", "N"]:
                    model.load_state_dict(torch.load(state_dict_file_name))
            except Exception as e:
                logging.exception(e)



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
deepdream_options = {
    "image_path": "images/Uil.jpg",
    "filter": 0,
    "use_gpu": True
}
lrp_options = {
    "image_path": "images/number.jpg",
    "use_MNIST": False,  # todo set to true
    "rho": "relu",
    "debug": True,  # todo set to false by default.
    "return_output": False,
    "use_gpu": True  # todo make an option for use_gpu!
}

#
# -----------------------------------------------------Sub-Menu---------------------------------------------------------
#

# LRP
def lrp_menu():
    _exit = False
    print("You can adjust the settings, see !help for more information.")
    while not _exit:
        _input = input("Type !start to start layerwise relevance propagation\n")
        if _input == "!help":
            print("The available commandos are:")
            print("!start: \t\tstart layerwise relevance propagation with the current options")
            print("!listOptions: \t\tlists all the current option values.")
            print("!exit: \t\t\texit layerwise relevance propagation.")
            print("You can adjust the next options:")
            for option in lrp_options:
                print("!", end="")
                print(option)
            print()
        elif _input == "!exit":
            print("layerwise relevance propagation won't be performed...")
            print()
            _exit = True
        elif _input == "!listOptions":
            for option in lrp_options:
                print(option, end="")
                print(":\t\t", end="")
                print(lrp_options.get(option))
            print()
        elif _input == "!image_path":
            lrp_options["image_path"] = input("Give the path to the image:\n")
            print()
        elif _input == "!rho":
            _input = input("Type the rho function you want to use (lin or relu):\n")
            if _input == "lin":
                lrp_options["rho"] = "lin"
            elif _input == "relu":
                lrp_options["rho"] = "relu"
            else:
                print("That is not an available option!")
        elif _input == "!use_gpu":  # todo set all similar functions in a method!!!!
            _input = input("Set the value for use_gpu, must be True or False:\n")
            if _input in ["True", "true", "t", "T"]:
                lrp_options["use_gpu"] = True
                print("use_gpu is set to:", end="")
                print(lrp_options.get("use_gpu"))
                print()
            elif _input in ["False", "false", "f", "F"]:
                lrp_options["use_gpu"] = False
                print("use_gpu is set to:", end="")
                print(lrp_options.get("use_gpu"))
                print()
            else:
                print("Not an available option...\n")

        elif _input == "!use_MNIST":
            _input = input("Do you want to use an image from the MNIST database?\n")
            if _input in ["Yes", "yes", "y", "Y"]:
                lrp_options["use_MNIST"] = True
                print("use_MNIST is set to:", end="")
                print(lrp_options.get("use_MNIST"))
                print()
            elif _input in ["No", "no", "n", "N"]:
                lrp_options["use_MNIST"] = False
                print("use_MNIST is set to:", end="")
                print(lrp_options.get("use_MNIST"))
                print()
            else:
                print("Not an available option...\n")
        elif _input == "!debug":
            _input = input("Do you want to debug?\n")
            if _input in ["Yes", "yes", "y", "Y"]:
                lrp_options["debug"] = True
                print("debug is set to:", end="")
                print(lrp_options.get("debug"))
                print()
            elif _input in ["No", "no", "n", "N"]:
                lrp_options["debug"] = False
                print("debug is set to:", end="")
                print(lrp_options.get("debug"))
                print()
            else:
                print("Not an available option...\n")
        elif _input == "!return_output":
            _input = input("Set the value for return_output, must be True or False:\n")
            if _input in ["True", "true", "t", "T"]:
                lrp_options["return_output"] = True
                print("return_output is set to:", end="")
                print(lrp_options.get("return_output"))
            elif _input in ["False", "false", "f", "F"]:
                lrp_options["return_output"] = False
                print("return_output is set to:", end="")
                print(lrp_options.get("return_output"))
            else:
                print("Not an available option...\n")
        elif _input == "!start":
            print("layerwise relevance propagation is running...")
            if modelType == "Convolutional":
                if lrp_options.get("use_MNIST"):
                    print("MNIST can't be used for convolutional neural networks.")
                image = Image.open(lrp_options.get("image_path"))
                relevance = convManager.layerwise_relevance_propagation(
                    image,
                    debug=lrp_options.get("debug"),
                    _return=lrp_options.get("return_output"),
                    rho=lrp_options.get("rho"))
            elif modelType == "Linear":
                if lrp_options.get("use_MNIST"):
                    for images, labels in test_loader:
                        relevance = mlpManager.layerwise_relevance_propagation(
                            images[0],
                            debug=lrp_options.get("debug"),
                            _return=lrp_options.get("return_output"),
                            rho=lrp_options.get("rho"))
                        break
                else:
                    image = Image.open(lrp_options.get("image_path")).convert('LA')
                    relevance = mlpManager.layerwise_relevance_propagation(
                        image,
                        debug=lrp_options.get("debug"),
                        _return=lrp_options.get("return_output"),
                        rho=lrp_options.get("rho"))
            plt.show()
        else:
            print("Not an available command, type !help to see all available commands.")



#Saliency
def saliency_menu():
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

# activmax
def activmax_menu():
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
                activmax_options["filters"] = int(_input)  # todo: make sure also multiple filters is possible.
        elif _input == "!conv_layer_int":
            _input = input("Give the number corresponding with the convolutional layer you want to see.\n")
            if _input == "None":
                activmax_options["conv_layer_int"] = None
            else:
                activmax_options["conv_layer_int"] = int(_input)
        elif _input == "!start":
            print("Activation maximisation is running...")
            convManager.activation_maximisation(use_gpu=activmax_options.get("use_gpu"),
                                                filters=activmax_options.get("filters"),
                                                last_layer=activmax_options.get("last_layer"),
                                                conv_layer_int=activmax_options.get("conv_layer_int"),
                                                return_output=activmax_options.get("return_output"))
            plt.show()
        else:
            print("Not an available command, type !help to see all available commands.")

def deepdream_menu():
    _exit = False
    while not _exit:
        _input = input("Type !start to start creating the deepdream\n")
        if _input == "!help":
            print("The available commandos are:")
            print("!start: \t\tstart the deepdream with the current options")
            print("!listOptions: \t\tlists all the current option values.")
            print("!exit: \t\t\texit deepdream.")
            print("You can adjust the next options:")
            for option in deepdream_options:
                print("!", end="")
                print(option)
            print()
        elif _input == "!exit":
            print("deepdream won't be created...")
            _exit = True
        elif _input == "!listOptions":
            for option in deepdream_options:
                print(option, end="")
                print(":\t\t", end="")
                print(deepdream_options.get(option))
            print()
        elif _input == "!image_path":
            deepdream_options["image_path"] = input("Give the path to the image:\n")
            print()
        elif _input == "!filter":
            deepdream_options["filter"] = int(input("Give the requested filter number:\n"))
            print()
        elif _input == "!use_gpu":
            _input = input("Set the value for use_gpu, must be True or False:\n")
            if _input in ["True", "true", "t", "T"]:
                deepdream_options["use_gpu"] = True
                print("use_gpu is set to:", end="")
                print(deepdream_options.get("use_gpu"))
                print()
            elif _input in ["False", "false", "f", "F"]:
                deepdream_options["use_gpu"] = False
                print("use_gpu is set to:", end="")
                print(deepdream_options.get("use_gpu"))
                print()
            else:
                print("Not an available option...\n")
        elif _input == "!start":
            print("Creating the deepdream...")
            convManager.deepdream(deepdream_options.get("image_path"),
                                  deepdream_options.get("filter"),
                                  use_gpu=deepdream_options.get("use_gpu"))
            plt.show()
        else:
            print("Not an available command, type !help to see all available commands.")



#
# -----------------------------------------------------Menu-------------------------------------------------------------
#

while running: # todo make sure that return values are used. check user input and images
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
                print("!saliency: \t\tuses the saliency explainability method.")
                print("!activmax: \t\tuses activation maximisation as explainability method.")
                print("!deepdream: \t\tcreate a deepdream from an image.")
                print("!lrp: \t\t\tuse the layerwise relevance propagation explainability method.")
            elif modelType == "Linear":
                print("\nThe multilayer perceptron specific commands you can do are:")
                print("!lrp: \t\t\tuse the layerwise relevance propagation explainability method.")
    elif _input == "!listModels":
        for tempModel in availableModels:
            print(tempModel)
    elif _input == "!addModel":
        file_name = input("Give the name of the python file (don't put .py behind the file!).\n")
        class_name = input("Give the name of the class.\n")
        try:
            exec('import %s' % file_name)
            availableModels.append("" + file_name + "." + class_name)
        except Exception as e:
            logging.exception(e)
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
            elif _input in availableModels:  # todo this is checked twice ==> remove one of both
                _exit = True
                load_model(_input)
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
            # yet again a copy of the model, otherwise you can get problems wit use_gpu functions
            convManager.set_model(model)
            saliency_menu()
        elif _input == "!activmax":
            print("You can change the requested options, type !help for more insight.")
            # yet again a copy of the model, otherwise you can get problems wit use_gpu functions
            convManager.set_model(model)
            activmax_menu()
        elif _input == "!deepdream":
            print("You can change the requested options, type !help for more insight.")
            # yet again a copy of the model, otherwise you can get problems wit use_gpu functions
            convManager.set_model(model)
            deepdream_menu()
        elif _input == "!lrp":
            convManager.set_model(model)
            lrp_menu()
        else:
            print("This is not an available command!")
    elif modelType == "Linear":
        if _input == "!lrp":
            # yet again a copy of the model, otherwise you can get problems wit use_gpu functions
            mlpManager.set_model(model)
            lrp_menu()
        else:
            print("This is not an available command!")
    else:
        print("This is not an available command!")
    print()


