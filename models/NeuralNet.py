import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=(128, 64), output_size=10):
        super(NeuralNet, self).__init__()

        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        # Hidden layer 1 to HL2 linear transformation
        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        # HL2 to output linear transformation
        self.layer3 = nn.Linear(hidden_size[1], output_size)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # HL1 with relu activation
        out = self.relu(self.layer1(x))
        # HL2 with relu activation
        out = self.relu(self.layer2(out))
        # Output layer with LogSoftmax activation
        out = self.Softmax(self.layer3(out))
        return out
