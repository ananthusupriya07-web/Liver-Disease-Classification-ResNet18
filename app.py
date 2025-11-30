# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 23:29:20 2025

@author: SUPRIYA ANANTHU
"""

# -*- coding: utf-8 -*-
"""
Streamlit Liver Disease Classifier App
"""

# -*- coding: utf-8 -*-
"""
Streamlit Liver Disease Classifier App
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights  # required for new PyTorch versions

# ---------------------------
# Load the trained ResNet18 model
# ---------------------------

# Load model with new PyTorch syntax (avoids deprecated "pretrained" argument)
model = models.resnet18(weights=None)

# Modify final layer to match your number of classes
model.fc = nn.Linear(model.fc.in_features, 3)

# Load trained weights
model.load_state_dict(torch.load("resnet18_model.pth", map_location=torch.device("cpu")))
model.eval()

# ---------------------------
# Image Preprocessing
# ---------------------------

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Prediction Function
# ---------------------------

def predict_image(image):
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Class mapping
class_names = {
    0: "CC",
    1: "NORMAL LIVER",
    2: "HCC"
}

# ---------------------------
# Streamlit User Interface
# ---------------------------

st.title("🧪 Liver Disease Classifier")
st.write("Upload a liver scan image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Updated to avoid deprecation warning
    st.image(image, caption="Uploaded Image", use_container_width=True)

    prediction = predict_image(image)
    
    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted Class: **{class_names[prediction]}**")
