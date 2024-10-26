import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from models.cataract_model import CataractClassificationModel
from utils.data_processing import CataractDataset, transform, load_image_paths_and_labels

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            labels = labels.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load the test data
    root_data_dir = './data/processed_images'
    test_image_paths, test_labels = load_image_paths_and_labels(root_data_dir, 'test')
    test_dataset = CataractDataset(test_image_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load the trained model
    model = CataractClassificationModel(num_classes=1)
    model.load_state_dict(torch.load('./models/cataract_classification_model.pth'))
    
    # Evaluate the model
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Cataract']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names=['Normal', 'Cataract'])

    # ROC curve
    plot_roc_curve(y_true, y_probs)

if __name__ == '__main__':
    main()
