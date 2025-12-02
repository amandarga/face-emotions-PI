# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Mapear classes para índices
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Carregar imagens
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_train_transforms_advanced(img_size):
    """Data augmentation AVANÇADO para melhor acurácia"""
    return A.Compose([
        # Transformações geométricas
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        
        # Ruído e blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        
        # Distorções
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3),
        ], p=0.2),
        
        # Cor e iluminação
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Sombras aleatórias
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 2),
            p=0.2
        ),
        
        # Dropout de patches
        A.CoarseDropout(
            max_holes=8,
            max_height=12,
            max_width=12,
            min_holes=4,
            fill_value=0,
            p=0.3
        ),
        
        # Resize e normalização
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(img_size):
    """Transformações de validação (sem augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])