import sys
from datetime import datetime
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim

from CNN_Architectures import CNNClassifier
from other_tools import get_model_information

from test_process import test_model
from train_process import train_model

from torchvision import models

########################################################################################################################
#                                                    USER PARAMETERS                                                   #
########################################################################################################################

dataset_path = "dataset"

results_path = "./results"

# Define the number of epochs of the model training
epoch_number = 3

# Define the size of the mini-batch
batch_size = 32

# Define the learning rate
learning_rate = 0.01

########################################################################################################################
#                                       USER ARG                                                                       #
########################################################################################################################
argv = sys.argv
model_name = None

if len(argv) > 1:
    model_name = argv[1]

if not model_name or model_name not in ["CNN", "VGG"]:
    print("Please provide a model name as an argument: CNN or VGG")
    sys.exit(1)

print(f"Model name: {model_name}")

results_path = results_path + "_" + model_name

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
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.Compose([
    transforms.Lambda(
        lambda y: torch.zeros(53, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
    )
])

"""" Train and validation set """
# Create the Python iterator for the train set (creating mini-batches)
train_set = torchvision.datasets.ImageFolder(dataset_path + "/train", transform=transform, target_transform=target_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# Create the Python iterator for the validation set
validation_set = torchvision.datasets.ImageFolder(dataset_path + "/valid", transform=transform, target_transform=target_transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
# Create the Python iterator for the test set
test_set = torchvision.datasets.ImageFolder(dataset_path + "/test", transform=transform, target_transform=target_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Load the first batch of images from the train set
images, labels = next(iter(train_loader))

# Get the shape of the images
image_shape = list(images.data.shape)
# Get automatically the number of channels of images
image_channel = image_shape[1]

# Get the number of classes from the dataset
classes = train_set.classes
class_number = len(list(classes))

########################################################################################################################
#                                  CHECK GPU AVAILABILITY AND CREATE THE NETWORK MODEL                                 #
########################################################################################################################

# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate and move the model to GPU
if model_name == "CNN":
    model = CNNClassifier(in_channel=image_channel, output_dim=class_number).to(device)
elif model_name == "VGG":
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, class_number)
    model = model.to(device)

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
