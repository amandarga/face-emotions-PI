import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys
from tqdm import tqdm

sys.path.append('src')
from models.emotion_model import EmotionCNN
from data.dataset import EmotionDataset, get_train_transforms_advanced, get_val_transforms
from training.mixup import mixup_data, mixup_criterion

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
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


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)

        return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))


class EmotionTrainer:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            self.config['emotion_recognition']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )

        print(f"Dispositivo: {self.device}")

        self.setup_data()
        self.create_model()

        self.best_acc = 0
        self.epochs_without_improvement = 0

    def setup_data(self):
        cfg = self.config['emotion_recognition']

        train_transform = get_train_transforms_advanced(cfg['img_size'])
        val_transform = get_val_transforms(cfg['img_size'])

        train_path = os.path.join(self.config['paths']['processed_data'], 'train')
        val_path = os.path.join(self.config['paths']['processed_data'], 'val')

        self.train_dataset = EmotionDataset(train_path, train_transform)
        self.val_dataset = EmotionDataset(val_path, val_transform)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['workers'],
            pin_memory=True
        )

        print(f"Treino: {len(self.train_dataset)} imagens")
        print(f"Validação: {len(self.val_dataset)} imagens")

    def create_model(self):
        cfg = self.config['emotion_recognition']

        self.model = EmotionCNN(
            num_classes=self.config['emotions']['num_classes'],
            backbone=cfg['backbone'],
            pretrained=cfg['pretrained'],
            dropout=cfg.get('dropout', 0.3)
        ).to(self.device)

        print(f"Modelo: {cfg['backbone']}")

        if cfg.get('label_smoothing', 0) > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                epsilon=cfg['label_smoothing']
            )
            print(f"Loss: Label Smoothing CE (ε={cfg['label_smoothing']})")
        else:
            self.criterion = FocalLoss(alpha=1, gamma=2)
            print(f"Loss: Focal Loss")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

        scheduler_cfg = cfg.get('scheduler', {})
        scheduler_type = scheduler_cfg.get('type', 'plateau')

        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg['epochs'],
                eta_min=scheduler_cfg.get('eta_min', 1e-6)
            )
            print(f"Scheduler: Cosine Annealing")
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,
                factor=0.5
            )
            print(f"Scheduler: ReduceLROnPlateau")

        self.scheduler_type = scheduler_type

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        cfg = self.config['emotion_recognition']
        use_mixup = cfg.get('use_mixup', False)
        mixup_prob = cfg.get('mixup_prob', 0.5)
        mixup_alpha = cfg.get('mixup_alpha', 0.2)

        pbar = tqdm(self.train_loader, desc=f'Época {epoch+1}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if use_mixup and torch.rand(1).item() < mixup_prob:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=mixup_alpha, device=self.device
                )

                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validação'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self):
        cfg = self.config['emotion_recognition']
        num_epochs = cfg['epochs']
        patience = cfg['patience']

        print(f"\nIniciando treinamento ({num_epochs} épocas)")
        print(f"Early stopping: {patience} épocas sem melhora\n")

        
        for epoch in range(num_epochs):
            # Treinar
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_acc = self.validate()
            
            # Atualizar scheduler
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_acc)
            
            # Log
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📊 Época {epoch+1}/{num_epochs}")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   LR: {current_lr:.6f}")
            
            # Salvar melhor modelo
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.epochs_without_improvement = 0
                
                save_path = os.path.join(
                    self.config['paths']['models'],
                    'best.pth'
                )
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }, save_path)
                
                print(f"   ✅ Melhor modelo salvo! (Val Acc: {val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
                print(f"   ⏳ Sem melhora por {self.epochs_without_improvement} épocas")
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\n⏹️  Early stopping! Melhor acc: {self.best_acc:.2f}%")
                break
        
        print(f"\n✅ Treinamento concluído!")
        print(f"🏆 Melhor acurácia: {self.best_acc:.2f}%")


if __name__ == "__main__":
    trainer = EmotionTrainer()
    trainer.train()