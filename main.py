from datetime import datetime
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import random_split

from CNN_Architectures import CNNClassifier
from VGG_Architectures import VGG16
from other_tools import get_model_information

from test_process import test_model
from train_process import train_model

########################################################################################################################
#                                                    USER PARAMETERS                                                   #
########################################################################################################################

dataset_path = "/home/matthieu/data/images_dataset"

results_path = "./results"

# Define the number of epochs of the model training
epoch_number = 3

# Define the size of the mini-batch
batch_size = 32

# Define the learning rate
learning_rate = 0.01


########################################################################################################################
#                                       CREATE A FOLDER AND FILE TO SAVE RESULTS                                       #
########################################################################################################################

# Get the date and time
now = datetime.now()
# Create the folder name
my_folder_name = now.strftime("%Y-%m-%d_%H" + "h" + "%M" + "min" + "%S" + "sec")
# Create the folder
os.makedirs(os.path.join(results_path, my_folder_name))
# Print a message in the console
print("\nResult folder created")

# Create and open a txt file to store information about the model performances
txt_file = open(os.path.join(os.path.join(results_path, my_folder_name), "Results.txt"), "a")

# Write information about hyperparameters
txt_file.write("Hyperparameters")
txt_file.write("\n")
txt_file.write("Epoch number: " + str(epoch_number))
txt_file.write("\n")
txt_file.write("Batch size: " + str(batch_size))
txt_file.write("\n")
txt_file.write("Learning rate: " + str(learning_rate))
txt_file.write("\n")


########################################################################################################################
#                                                LOAD THE DATASET                                                #
########################################################################################################################

""" Data Transformation """
transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))])

"""" Train and validation set """
# Load the dataset by applying transformations
dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform, target_transform=target_transform)

# Divide the dataset into a train, validation and test sets
generator1 = torch.Generator().manual_seed(42)
train_set, validation_set, test_set = random_split(dataset,[0.7, 0.15, 0.15], generator=generator1)

# Create the Python iterator for the train set (creating mini-batches)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# Create the Python iterator for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
# Create the Python iterator for the test set
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Load the first batch of images from the train set
images, labels = next(iter(train_loader))

# Get the shape of the images
image_shape = list(images.data.shape)
# Get automatically the number of channels of images
image_channel = image_shape[1]

# Get the number of classes from the dataset
classes = dataset.classes
class_number = len(list(classes))

########################################################################################################################
#                                  CHECK GPU AVAILABILITY AND CREATE THE NETWORK MODEL                                 #
########################################################################################################################

# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate and move the model to GPU
# model = MLPClassier(input_size=784, hidden_size=128, output_size=10).to(device)
# model = CNNClassifier(in_channel=image_channel, output_dim=class_number).to(device)
model = VGG16(in_channel=image_channel, output_dim=class_number).to(device)

# Print information about the model
get_model_information(model, txt_file)


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

# Train the neural network
model = train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device)
# Test the neural network
test_model(test_loader, model, loss_function, device, classes, txt_file)
# Save the model
torch.save(model.state_dict(), "model.pth")

# Close your txt file
txt_file.close()
