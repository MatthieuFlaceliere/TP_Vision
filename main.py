import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import random_split
from other_tools import get_model_information


from MLP_Architectures import MLPClassier
from test_process import test_model
from train_process import train_model

########################################################################################################################
#                                                    USER PARAMETERS                                                   #
########################################################################################################################

# Define the number of epochs of the model training
epoch_number = 5

# Define the size of the mini-batch
batch_size = 4

# Define the learning rate
learning_rate = 0.01

########################################################################################################################
#                                                LOAD THE MNIST DATASET                                                #
########################################################################################################################

""" Data Transformation """
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: torch.flatten(x))])
target_transform = transforms.Compose([transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))])

"""" Train and validation set """
train_set = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform,target_transform=target_transform)

train_set_shape = list(train_set.data.shape)
train_set, validation_set = random_split(train_set,(int(train_set_shape[0]*0.9),train_set_shape[0]-int(train_set_shape[0]*0.9)))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch_size, shuffle=False)

""" Test set """
test_set = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform,target_transform=target_transform)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True)

########################################################################################################################
#                                  CHECK GPU AVAILABILITY AND CREATE THE NETWORK MODEL                                 #
########################################################################################################################

# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate and move the model to GPU
model = MLPClassier(input_size=784, hidden_size=128, output_size=10).to(device)

# Print information about the model
get_model_information(model)


########################################################################################################################
#                                          SET THE LOSS FUNCTION AND OPTIMIZER                                         #
########################################################################################################################

# Create the loss function
loss_function = nn.CrossEntropyLoss()
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

########################################################################################################################
#                                           TRAIN AND TEST THE NETWORK MODEL                                           #
########################################################################################################################

# Check if model.pth file exists
if os.path.isfile("model.pth"):
    # Load the model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("Model loaded successfully")
else:
    # Train the neural network
    model = train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device)
    # Test the neural network
    test_model(test_loader, model, loss_function, device)
    # Save the model
    torch.save(model.state_dict(), "model.pth")

# Try model in real time
# Load the model
model = MLPClassier(input_size=784, hidden_size=128, output_size=10).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()
print("Model loaded successfully")

