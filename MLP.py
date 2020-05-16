import ExplainabilityMethods.LRP as LRP
class MLP:
    """This class will provide explainability methods from other classes.

    This class will only contain explainability methods that are compatible with MLP networks.
    In every method call, we will copy the model to perform an explainability method to.
    We do this to make sure that the forward and backward hooks won't be saved on the model and the original model is
    not adjusted by the explainability.

    Arg:
        model : A multilayer perceptron model.
    """
    def __init__(self):
        self.model = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def get_model(self):
        return self.model

    def set_model(self, model):
        """
        This method will set the class model variable.
        :param model: Must be a MLP model with only ReLU and Linear layers and layers like SoftMax.
                      The input layer of this model must be one compatible with a square image f.e. 100 x 100.
                      If the input layer has a number of neurons that has not an integer as root, LRP will not work!
        :return:
        """
        self.model = model  # make sure that you won't adjust the original model by registering hooks.

    def layerwise_relevance_propagation(self, _input, debug=False, _return=False, rho="lin"):
        """
        Perform a layerwise relevance propagation on an _input image with the model that needed to be set previously
        with the set_model method.
        :param _input: an input tensor of a in instance of PIL.Image.Image. This will be rescaled automatically.
        :param debug:   (bool) True if you want debug statements printed in the terminal.
        :param _return: (bool) True if the relevance values need to be returned.
        :param rho:     (String) 'lin' if you want a linear rho function, 'relu' if you want to use a relu function.
                        This function will be used for all Linear layers.
        :return:        The calculated relevances for a specific input image.
        """
        # the copy is already done in LRP so it is not necessary here.
        _layerwise_relevance_propagation = LRP.LRP()
        _layerwise_relevance_propagation.lrp(self.model, _input, debug=debug, _return=_return, rho=rho, model_type="MLP")
