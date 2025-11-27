import torch
import torch.nn as nn
import timm

class EmotionCNN(nn.Module):
    """Modelo para classificação de emoções"""
    
    def __init__(self, num_classes=7, backbone='efficientnet_b0', pretrained=True):
        super(EmotionCNN, self).__init__()
        
        # Backbone pré-treinado
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0  # Remove classificador
        )
        
        # Obter dimensão de features
        num_features = self.backbone.num_features
        
        # Classificador customizado
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class FocalLoss(nn.Module):
    """Focal Loss para lidar com classes desbalanceadas"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss