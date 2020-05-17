# Explainable_Neural_Network
## goal of this project
This project is meant to become a toolbox to compare all kinds of neural networks with each other with different methods.
By doing so we can compare a Multilayer Perceptron network with (MLP) a Convolutional Neural Network (CNN) with f.e. layerwise relevance propagation. There are also methods that only work for one specific type of network f.e. CNN. These methods can be used to compare multiple networks of that type. 

## project structure
This project consist of multiple .py and .pt files. Furthermore, there are also a lot of directories. 
That's why I will explain what you will find in there in this chapter.
### Root folder
in here you will find the main.py, Convolutional.py, MLP.py and ModelFunctions.py files. 

The main.py is the main function and will start the toolbox in comment line. After starting up, 
there will be a UI so the user can give commands and this program will execute these.
If you select a loaded neural network, this main function will call a method from ModelFunction.py that
checks which kind of model is selected. Depending on this, the methods for a CNN or MLP will be available. 
Once an explainability method is show and the user types the command "!start", the corresponding message will be ran 
in MLP.py of Convolutional.py (depending on the type of network).

The MLP.py file contains the MLP class and will be able to run all methods from the files in the folder ExplainabilityMethods that are compatible with MLP networks. 
At this moment, only LRP and saliency is available. When a user type a command f.e. "!lrp", this class will call the corresponding method in the LRP.py file,
but with the correct values so it will work for a MLP network.

The Convolutional.py file contains the Convolutional class and works about the same as the MLP class, but with the difference that
 this class can run all methods that are compatible with CNN.

The ModelFunctions.py file will contain a lot of functions that can be used in all classes once imported. 

### ExplainabilityMethods
This folder will contain the actual implementations of the explainability methods, that will be called by either the MLP class or Convolutional class.

The LRP file contains the LRP class and will perform Layerwise Relevance Propagation through all layers of the module.

The Flashtorch directory is created by Misa Ogura and all info can be found at https://github.com/MisaOgura/flashtorch.
This will contain the saliency, activation maximisation and deepdream explainability implementations. If you compare 
the GitHub code from Misa Ogure and mine, you'll see that I made adjustments to generalise this code.

### MNIST
here you can find the MNIST Handwritten Digit Classification Dataset.

### models
Here you can find self created and trained models. These are not state of the art models, but it's purpose is mainly to demonstrate
that this toolkit works for all kinds of models and that you can compare models to see which model is best. 
The .pt files contain the state_dict of the models. Here is a list which one belongs to which model:

mnist_model.pt          ==> NeuralNet.py (with input size 28*28) 

mnist_model_29x29.pt    ==> NeuralNet.py (with input size 29*29)

mnist_model_seq.pt      ==> NeuralNetSeq.py

TestNet.pt              ==> TestNet.py

## get started:
when you run this program, you can add a model with the command "!addModel", where you need to type f.e. "models.TestNet" for selecting the
.py file and "TestNet" for the class Name. once the model is added to the program, you can select this model with 
"!selectModel". you'll find what you need to type by using the command "!list". In our case it will be "models.TestNet.TestNet"
. For the path to the state_dict, you can type "models/TestNet.pt" in our example.

you can of coarse add/select your own models as well, when you have the .py and .pt file.

## disclaimer
This program is still not fully finished and a lot of improvements can be made in the future. 

1) At this moment LRP cannot be executed with models that contains AvgPool or AdaptiveAvgPool layers. This means that LRP can't be used
for the state of the art networks like alexnet or vgg16.

2) The main menu and smaller menus need to be cleaned up a bit and some functionalities are not yet implemented.

3) Only CNN and MLP networks are supported at this moment. RNN and so forth are not implemented yet.

4) The MLP networks must have an input size of a square f.e. 28x28 is fine, but 102 x 120 will not work.





