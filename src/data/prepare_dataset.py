import os
import yaml
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def prepare_emotion_dataset(config_path='configs/config.yaml'):
    """
    Prepara dataset de emoções a partir de estrutura de pastas
    
    Estrutura esperada:
    data/raw/emotions/
        ├── feliz/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── triste/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── ...
    """
    
    # Carregar config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    emotions = config['emotions']['classes']
    
    emotion_path = os.path.join(raw_path, 'emotions')
    
    # Coletar imagens e labels
    image_paths = []
    labels = []
    
    print("📂 Coletando imagens...")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(emotion_path, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Diretório não encontrado: {emotion_dir}")
            continue
        
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_dir, img_name)
                image_paths.append(img_path)
                labels.append(emotion_idx)
    
    print(f"\n✅ Total de imagens: {len(image_paths)}")
    print(f"📊 Distribuição:")
    label_counts = Counter(labels)
    for emotion_idx, count in sorted(label_counts.items()):
        print(f"   {emotions[emotion_idx]}: {count}")
    
    # Split train/val/test
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        train_size=train_split,
        random_state=config['random_seed'],
        stratify=labels
    )
    
    val_ratio = val_split / (val_split + config['data']['test_split'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=config['random_seed'],
        stratify=y_temp
    )
    
    print(f"\n📊 Divisão dos dados:")
    print(f"   Treino: {len(X_train)} ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"   Validação: {len(X_val)} ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"   Teste: {len(X_test)} ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    # Salvar splits
    splits_dir = os.path.join(processed_path, 'emotions')
    os.makedirs(splits_dir, exist_ok=True)
    
    splits = {
        'train': {'paths': X_train, 'labels': y_train},
        'val': {'paths': X_val, 'labels': y_val},
        'test': {'paths': X_test, 'labels': y_test}
    }
    
    for split_name, split_data in splits.items():
        split_file = os.path.join(splits_dir, f'{split_name}.json')
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"✅ Salvo: {split_file}")
    
    print("\n✅ Preparação concluída!")
    return splits

def prepare_detection_dataset_yolo(annotations_dir, output_dir):
    """
    Prepara dataset para YOLOv8
    
    Formato de anotação esperado (YOLO):
    - Um arquivo .txt para cada imagem
    - Formato: class x_center y_center width height (normalizados 0-1)
    """
    print("🎯 Preparando dataset de detecção para YOLOv8...")
    
    # Criar estrutura YOLO
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    print(f"✅ Estrutura criada em: {output_dir}")
    print("\n📝 Para anotar suas imagens, use:")
    print("   - LabelImg: https://github.com/heartexlabs/labelImg")
    print("   - Roboflow: https://roboflow.com")
    print("   - CVAT: https://cvat.org")

if __name__ == "__main__":
    # Preparar dataset de emoções
    prepare_emotion_dataset()
    
    # Preparar dataset de detecção
    prepare_detection_dataset_yolo(
        'data/raw/annotations',
        'data/processed/detection'
    )