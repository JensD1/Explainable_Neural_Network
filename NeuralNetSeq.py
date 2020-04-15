import torch.nn as nn


class NeuralNetSeq(nn.Module):
    def __init__(self):
        super(NeuralNetSeq, self).__init__()
        # layers
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # dit moet naargelang het aantal layers en type neuraal netwerk anders forwarden.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
