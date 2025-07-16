# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 00:16:58 2025

@author: Snehal
"""
import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model class (must match training code)
from torchvision.models import densenet121
model = densenet121(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.classifier.in_features, 3)  # adjust for your 3 classes
)
model.load_state_dict(torch.load("model/best_densenet_model.pth", map_location=device))
model.eval().to(device)

class_names = ["Covid19", "Viral Pneumonia", "NORMAL"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]
