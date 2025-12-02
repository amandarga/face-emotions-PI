import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

sys.path.append('src')
from models.emotion_model import EmotionCNN
from data.dataset import EmotionDataset, get_val_transforms

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("\n" + "="*60)
    print("📦 INFORMAÇÕES DO CHECKPOINT")
    print("="*60)
    print(f"Época treinada: {checkpoint.get('epoch', 'N/A')}")
    print(f"Acurácia de Validação: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"Acurácia de Treino: {checkpoint.get('train_acc', 'N/A'):.2f}%")
    print("="*60 + "\n")
    return checkpoint

def evaluate_model(model, dataloader, device, emotion_classes):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Avaliando'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'precision_per_class': precision_per_class * 100,
        'recall_per_class': recall_per_class * 100,
        'f1_per_class': f1_per_class * 100,
        'support_per_class': support_per_class,
        'confusion_matrix': cm
    }

def main():
    # Carregar configurações
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}\n")
    
    # Carregar checkpoint
    checkpoint_path = os.path.join(config['paths']['models'], 'best.pth')
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Criar modelo
    model = EmotionCNN(
        num_classes=config['emotions']['num_classes'],
        backbone=config['emotion_recognition']['backbone'],
        pretrained=False,
        dropout=config['emotion_recognition'].get('dropout', 0.3)
    ).to(device)
    
    # Carregar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Modelo carregado: {config['emotion_recognition']['backbone']}\n")
    
    # Preparar dados de teste
    val_transform = get_val_transforms(config['emotion_recognition']['img_size'])
    
    # Tentar carregar conjunto de teste, se não existir, usar validação
    test_path = os.path.join(config['paths']['processed_data'], 'test')
    if os.path.exists(test_path):
        test_dataset = EmotionDataset(test_path, val_transform)
        dataset_name = "TESTE"
    else:
        test_path = os.path.join(config['paths']['processed_data'], 'val')
        test_dataset = EmotionDataset(test_path, val_transform)
        dataset_name = "VALIDAÇÃO"
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 Avaliando no conjunto de {dataset_name}: {len(test_dataset)} imagens\n")
    
    # Avaliar
    emotion_classes = config['emotions']['classes']
    metrics = evaluate_model(model, test_loader, device, emotion_classes)
    
    # Imprimir resultados
    print("\n" + "="*60)
    print(f"📊 RESULTADOS DA AVALIAÇÃO (Conjunto de {dataset_name})")
    print("="*60)
    print(f"Acurácia Geral: {metrics['accuracy']:.2f}%")
    print(f"Precisão Média: {metrics['precision']:.2f}%")
    print(f"Recall Médio: {metrics['recall']:.2f}%")
    print(f"F1-Score Médio: {metrics['f1']:.2f}%")
    print("="*60)
    
    print("\n📊 MÉTRICAS POR EMOÇÃO:")
    print("-" * 70)
    print(f"{'Emoção':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12} {'Amostras':<10}")
    print("-" * 70)
    
    for i, emotion in enumerate(emotion_classes):
        print(f"{emotion:<12} {metrics['precision_per_class'][i]:>10.2f}% "
              f"{metrics['recall_per_class'][i]:>10.2f}% "
              f"{metrics['f1_per_class'][i]:>10.2f}% "
              f"{int(metrics['support_per_class'][i]):>10}")
    
    print("-" * 70)
    
    # Matriz de confusão
    print("\n📊 MATRIZ DE CONFUSÃO:")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    
    # Cabeçalho
    header = "Real \\ Pred".ljust(12)
    for emotion in emotion_classes:
        header += emotion[:8].center(10)
    print(header)
    print("-" * 70)
    
    # Linhas
    for i, emotion in enumerate(emotion_classes):
        row = emotion[:10].ljust(12)
        for j in range(len(emotion_classes)):
            row += str(cm[i][j]).center(10)
        print(row)
    
    print("="*60 + "\n")
    
    # Salvar relatório
    report_path = os.path.join(config['paths']['models'], 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO DE AVALIAÇÃO - {dataset_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Acurácia Geral: {metrics['accuracy']:.2f}%\n")
        f.write(f"Precisão Média: {metrics['precision']:.2f}%\n")
        f.write(f"Recall Médio: {metrics['recall']:.2f}%\n")
        f.write(f"F1-Score Médio: {metrics['f1']:.2f}%\n\n")
        f.write("MÉTRICAS POR EMOÇÃO:\n")
        f.write("-" * 60 + "\n")
        for i, emotion in enumerate(emotion_classes):
            f.write(f"{emotion}: P={metrics['precision_per_class'][i]:.2f}% "
                   f"R={metrics['recall_per_class'][i]:.2f}% "
                   f"F1={metrics['f1_per_class'][i]:.2f}%\n")
    
    print(f"📄 Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()