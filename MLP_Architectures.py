import torch.nn as nn

# Define the MLP architecture for the classifier
class MLPClassier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.act_output = nn.Softmax(dim=1)

    def forward(self, x):
        z1 = self.hidden1(x)
        a1 = self.act1(z1)
        z2 = self.hidden2(a1)
        a2 = self.act2(z2)
        z3 = self.output(a2)
        y = self.act_output(z3)
        return y