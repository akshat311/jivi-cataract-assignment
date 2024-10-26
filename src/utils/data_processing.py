import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

def load_image_paths_and_labels(root_dir, split):
    image_paths = []
    labels = []

    class_to_label = {'cataract': 1, 'normal': 0}
    
    for class_name in ['cataract', 'normal']:
        class_dir = os.path.join(root_dir, split, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_dir, filename)
                image_paths.append(file_path)
                labels.append(class_to_label[class_name])

    return image_paths, labels
