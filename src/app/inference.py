import torch
from torchvision import transforms
from PIL import Image
from models.cataract_model import CataractClassificationModel

model_path = './models/cataract_classification_model.pth'

def load_model():
    model = CataractClassificationModel(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        predicted_class = 'cataract' if confidence > 0.5 else 'normal'
        return predicted_class, confidence
