import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from CNN_Surface_detect.model import Surface_Defect_Detection

st.set_page_config(page_title="Surface Defect AI", layout="wide")


# LOAD MODEL
@st.cache_resource
def load_model():
    model = Surface_Defect_Detection(num_classes=6)
    model.load_state_dict(torch.load("models/surface_defect_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# GRAD CAM
def generate_gradcam(model, image_tensor, class_idx):

    gradients = []
    activations = []

    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[0, class_idx]

    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().numpy()

    cam = cv2.resize(cam, (200,200))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    fh.remove()
    bh.remove()

    return heatmap



# PREDICTION FUNCTION
def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    heatmap = generate_gradcam(model, img, pred.item())
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return classes[pred.item()], conf.item()*100, heatmap


# UI

st.title("🏭 Surface Defect Detection System")

option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Click Photo"]
)

#  UPLOAD IMAGE

if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload Surface Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        pred, conf, heatmap = predict(image)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

        st.info(f"Confidence: {conf:.2f}%")

        if conf < 65.0:
            st.success("✅ Normal Surface — No Defect Detected (Low Confidence, Surface is OK)")
        else:
            st.error(f"⚠️ Defect Detected: **{pred}** (Confidence: {conf:.2f}%)")



#  CLICK PHOTO (Cloud Compatible)

elif option == "Click Photo":

    camera_image = st.camera_input("Take a Photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        pred, conf, heatmap = predict(image)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Captured Image", use_column_width=True)
        col2.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

        st.info(f"Confidence: {conf:.2f}%")

        if conf < 65.0:
            st.success("✅ Normal Surface — No Defect Detected (Low Confidence, Surface is OK)")
        else:
            st.error(f"⚠️ Defect Detected: **{pred}** (Confidence: {conf:.2f}%)")