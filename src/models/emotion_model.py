# src/models/emotion_model.py
import torch
import torch.nn as nn
import timm

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, backbone='efficientnet_b0', pretrained=True, dropout=0.3):
        super().__init__()
        
        # Carregar backbone pré-treinado
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        
        # Obter número de features
        if 'efficientnet' in backbone:
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in backbone:
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            num_features = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
        
        # Cabeça de classificação com dropout configurável
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # NOVO: dropout configurável
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),  # NOVO: dropout configurável
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out