import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from model import Surface_Defect_Detection
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

transform=transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data=datasets.ImageFolder("neu-surface-defect-database/NEU-DET/train/images",transform=transform)
test_data=datasets.ImageFolder("neu-surface-defect-database/NEU-DET/validation/images",transform=transform)

train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32,shuffle=False)

class1=train_data.classes
num_classes=len(class1)
print("Num-Classes:-",num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model=Surface_Defect_Detection().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

from tqdm import tqdm
epochs=30
for epoch in range(epochs):
    model.train()
    running_loss=0
    for img,label in tqdm(train_loader):
        img , label = img.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(img)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct=0
total=0

with torch.no_grad():
    for img,label in test_loader:
        img , label = img.to(device),label.to(device)
        outputs=model(img)
        _,pred=torch.max(outputs.data,1)
        total += label.size(0)
        correct += (pred == label).sum().item()
print("Test Accuracy:", 100 * correct / total)


torch.save(model.state_dict(), "surface_defect_model.pth")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)          
        label = label.to(device)      

        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())   
        all_labels.extend(label.cpu().numpy())



print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']))




cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()