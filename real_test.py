import os

import cv2
import numpy as np
import torch
from torchviz import make_dot

from CNN_Architectures import CNNClassifier

# Try model in real time
# Load the model
model = CNNClassifier(in_channel=3, output_dim=7).to("cpu")
if os.path.isfile("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("Model loaded successfully")
else:
    print("Model not found")
    exit()