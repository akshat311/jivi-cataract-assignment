import torch
from torch.utils.data import DataLoader
from models.cataract_model import CataractClassificationModel
from utils.data_processing import load_image_paths_and_labels, CataractDataset
from utils.training import train_model, get_optimizer, get_criterion
from torchvision import transforms

root_data_dir = '../data/processed_images'
train_image_paths, train_labels = load_image_paths_and_labels(root_data_dir, 'train')
test_image_paths, test_labels = load_image_paths_and_labels(root_data_dir, 'test')

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    
    train_dataset = CataractDataset(train_image_paths, train_labels, transform=transform_train)
    test_dataset = CataractDataset(test_image_paths, test_labels, transform=transform_test)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True),
        'val': DataLoader(test_dataset, batch_size=4, shuffle=False)
    }

    model = CataractClassificationModel(num_classes=1)
    optimizer = get_optimizer(model)
    criterion = get_criterion()

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=30)

    torch.save(trained_model.state_dict(), './models/cataract_classification_model.pth')
