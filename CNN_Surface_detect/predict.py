from PIL import Image
import torch
from torchvision import transforms
from model import Surface_Defect_Detection

def predict_image(image_path):
    image=Image.open(image_path)
    transform=transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
    image=transform(image).unsqueeze(0)
    model=Surface_Defect_Detection()
    model.load_state_dict(torch.load("models/surface_defect_model.pth"))
    model.eval()

    outputs=model(image)
    _,preds=torch.max(outputs,1)

    classes=6
    print("Prediction:", classes[preds.item()])
