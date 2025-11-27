import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EmotionDataset(Dataset):
    """Dataset para treinamento de reconhecimento de emoções"""
    
    def __init__(self, image_paths, labels, transform=None, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformações
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label

def get_train_transforms(img_size=224):
    """Augmentations para treinamento"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """Transformações para validação"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])