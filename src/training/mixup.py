# src/training/mixup.py
"""
Mixup Data Augmentation
Mistura duas imagens para criar exemplos sintéticos
Reduz overfitting significativamente
"""

import numpy as np
import torch

def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Aplica Mixup augmentation
    
    Parâmetros:
    -----------
    x : tensor
        Batch de imagens [batch_size, 3, H, W]
    y : tensor
        Labels [batch_size]
    alpha : float
        Parâmetro da distribuição Beta (padrão: 0.2)
    
    Retorna:
    --------
    mixed_x : tensor
        Imagens mixadas
    y_a, y_b : tensor
        Labels originais
    lam : float
        Lambda (peso da mistura)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Mistura as imagens: mixed = λ * img_a + (1-λ) * img_b
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss function para Mixup
    Loss = λ * loss(pred, y_a) + (1-λ) * loss(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix: corta um patch de uma imagem e cola em outra
    Alternativa ao Mixup, às vezes mais efetivo
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]
    
    # Coordenadas do patch
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Corta e cola
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Ajusta lambda baseado no tamanho do patch
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Gera coordenadas aleatórias para o patch do CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2