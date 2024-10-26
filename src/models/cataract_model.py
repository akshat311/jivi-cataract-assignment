import torch.nn as nn
from torchvision.models import vit_b_16

class CataractClassificationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CataractClassificationModel, self).__init__()
        # Load pre-trained ViT model
        self.encoder = vit_b_16(pretrained=True)
        
        # Freeze the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Replace the original head with a custom classifier for binary classification
        self.encoder.heads = nn.Sequential(
            nn.Linear(self.encoder.heads.head.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        return self.encoder(x)
