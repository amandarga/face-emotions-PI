import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('src')
from data.dataset import EmotionDataset, get_train_transforms, get_val_transforms
from models.emotion_model import EmotionCNN, FocalLoss

class EmotionTrainer:
    def __init__(self, config_path='configs/config.yaml'):
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['emotion_recognition']['device']
            if torch.cuda.is_available() else 'cpu'
        )
        print(f"🖥️  Usando device: {self.device}")
        
        # Criar diretórios
        self.checkpoint_dir = 'models/checkpoints/emotions'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter('runs/emotion_training')
        
        # Carregar dados
        self.load_data()
        
        # Criar modelo
        self.create_model()
        
        # Métricas
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def load_data(self):
        """Carregar datasets"""
        print("📂 Carregando dados...")
        
        splits_dir = os.path.join(
            self.config['data']['processed_path'],
            'emotions'
        )
        
        # Carregar splits
        with open(os.path.join(splits_dir, 'train.json'), 'r') as f:
            train_data = json.load(f)
        with open(os.path.join(splits_dir, 'val.json'), 'r') as f:
            val_data = json.load(f)
        
        img_size = self.config['emotion_recognition']['img_size']
        
        # Criar datasets
        train_dataset = EmotionDataset(
            train_data['paths'],
            train_data['labels'],
            transform=get_train_transforms(img_size)
        )
        
        val_dataset = EmotionDataset(
            val_data['paths'],
            val_data['labels'],
            transform=get_val_transforms(img_size)
        )
        
        # DataLoaders
        batch_size = self.config['emotion_recognition']['batch_size']
        workers = self.config['emotion_recognition']['workers']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )
        
        print(f"✅ Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")
    
    def create_model(self):
        """Criar modelo e otimizador"""
        cfg = self.config['emotion_recognition']
        
        self.model = EmotionCNN(
            num_classes=self.config['emotions']['num_classes'],
            backbone=cfg['backbone'],
            pretrained=cfg['pretrained']
        ).to(self.device)
        
        print(f"🧠 Modelo criado: {cfg['backbone']}")
        
        # Loss function
        self.criterion = FocalLoss(alpha=1, gamma=2)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=5,
            factor=0.5
        )
    
    def train_epoch(self, epoch):
        """Treinar uma época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch+1}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Atualizar progresso
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': running_loss/len(pbar),
                'acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validar modelo"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Salvar checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Salvar último checkpoint
        path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint, path)
        
        # Salvar melhor modelo
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, path)
            print(f"💾 Melhor modelo salvo! Acc: {val_acc:.2f}%")
    
    def plot_confusion_matrix(self, labels, predictions, epoch):
        """Plotar matriz de confusão"""
        emotions = self.config['emotions']['classes']
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
        plt.title(f'Matriz de Confusão - Época {epoch+1}')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        
        save_path = os.path.join(self.checkpoint_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        plt.close()
    
    def train(self):
        """Loop principal de treinamento"""
        epochs = self.config['emotion_recognition']['epochs']
        patience = self.config['emotion_recognition']['patience']
        
        print(f"\n🚀 Iniciando treinamento por {epochs} épocas...")
        print(f"⏰ Early stopping: {patience} épocas sem melhora\n")
        
        for epoch in range(epochs):
            # Treinar
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_acc, preds, labels = self.validate()
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            print(f"\nÉpoca {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Scheduler
            self.scheduler.step(val_acc)
            
            # Salvar checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Plotar matriz de confusão a cada 10 épocas
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(labels, preds, epoch)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⏹️  Early stopping! Sem melhora por {patience} épocas")
                break
        
        print(f"\n✅ Treinamento concluído!")
        print(f"🏆 Melhor acurácia: {self.best_val_acc:.2f}%")
        
        self.writer.close()

if __name__ == "__main__":
    trainer = EmotionTrainer()
    trainer.train()