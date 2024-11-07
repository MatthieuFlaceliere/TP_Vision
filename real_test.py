import os

import cv2
import numpy as np
import torch

from CNN_Architectures import CNNClassifier

# Try model in real time
# Load the model
model = CNNClassifier(in_channel=3, output_dim=10).to("cpu")
if os.path.isfile("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("Model loaded successfully")
else:
    print("Model not found")
    exit()

def draw_digit():
    # Créer une image noire
    img = np.zeros((300, 300), dtype=np.uint8)
    # Créer une fenêtre
    cv2.namedWindow("Digit")
    drawing = False  # Variable pour suivre si on dessine ou pas

    # Fonction de rappel pour dessiner
    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:  # Si le bouton gauche est enfoncé
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:  # Si la souris se déplace
            if drawing:
                cv2.circle(img, (x, y), 10, (255), -1)  # Dessine un cercle blanc
        elif event == cv2.EVENT_LBUTTONUP:  # Si le bouton gauche est relâché
            drawing = False

    # Associer la fenêtre avec la fonction de rappel
    cv2.setMouseCallback("Digit", draw)

    # Afficher la fenêtre
    while True:
        cv2.imshow("Digit", img)
        key = cv2.waitKey(1)
        if key == ord("q"):  # Quitte quand on appuie sur "q"
            break

    # Fermer la fenêtre
    cv2.destroyAllWindows()
    return img


img = draw_digit()
img = cv2.resize(img, (28, 28))  # Redimensionner l'image en 28x28
cv2.imshow("Digit resized", img)
cv2.waitKey(0)
img = np.reshape(img, (1, 1, 28, 28))  # Adapter la forme pour [batch, channels, height, width]
img = torch.from_numpy(img).float() / 255.0  # Convertir en float et normaliser
img = img.to("cpu")

output = model(img.to("cpu"))
_, predicted = torch.max(output.data, 1)
print("Predicted digit:", predicted.item())

cv2.destroyAllWindows()