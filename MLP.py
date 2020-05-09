import ExplainabilityMethods.LRP as LRP
class MLP:
    def __init__(self):
        self.model = None

    #
    # --------------------------------------------------Methods---------------------------------------------------------
    #
    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model  # make sure that you won't adjust the original model by registering hooks.

    def layerwise_relevance_propagation(self, _input, debug=False, _return=False, rho="lin"):
        _layerwise_relevance_propagation = LRP.LRP()
        _layerwise_relevance_propagation.lrp(self.model, _input, debug=debug, _return=_return, rho=rho, model_type="MLP")
