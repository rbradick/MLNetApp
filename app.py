
import os
os.environ["STREAMLIT_HOME"] = "/tmp"
os.environ["STREAMLIT_TELEMETRY_ENABLED"] = "0"
os.environ["STREAMLIT_SUPPRESS_CONFIG_WARNINGS"] = "true"

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import requests

class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.deconv = nn.ConvTranspose2d(512, 1, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.deconv(x)
        return x

@st.cache_resource
def load_model():
    model_path = "mlnet.pth"
    if not os.path.exists(model_path):
        with requests.get("https://huggingface.co/rbradick76/mlnet-saliency/resolve/main/mlnet.pth", stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = MLNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.set_page_config(page_title="MLNet Saliency Heatmap", layout="wide")
st.title("🧠 MLNet-based Saliency Heatmap Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        MAX_SIZE = 512
        if max(image.size) > MAX_SIZE:
            scale = MAX_SIZE / max(image.size)
            new_size = tuple([int(dim * scale) for dim in image.size])
            image = image.resize(new_size)

        image_np = np.array(image)
        input_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(input_tensor)
            saliency = output.squeeze().numpy()
            saliency = cv2.GaussianBlur(saliency, (11, 11), 0)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Original Image", use_container_width=True)
        with col2:
            fig, ax = plt.subplots()
            ax.imshow(image_np)
            ax.imshow(saliency, cmap='jet', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("⬆️ Upload an image file to generate a saliency heatmap.")
