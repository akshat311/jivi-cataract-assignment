import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, save_path="./models/best_cataract_classification_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())  # Track the best model weights
    best_val_accuracy = 0.0  # Track the best validation accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()  # Binary prediction
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("Best validation accuracy improved. Model saved.")

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model to the specified path
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    return model

def get_optimizer(model, lr=1e-4):
    return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def get_criterion():
    return nn.BCELoss()
