# src/data/prepare_dataset.py
import os
import json
import yaml
import shutil
from tqdm import tqdm
from pathlib import Path

def convert_json_to_folders():
    """
    Converte estrutura antiga (train.json) para nova (pastas)
    """
    
    # Carregar configuração
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    processed_path = config['paths']['processed_data']
    emotion_names = config['emotions']['classes']
    
    # Mapeamento de índices para nomes com prefixo
    emotion_map = {
        0: '0_raiva',
        1: '1_nojo',
        2: '2_medo',
        3: '3_feliz',
        4: '4_neutro',
        5: '5_triste',
        6: '6_surpresa'
    }
    
    print("🔄 Convertendo JSONs para estrutura de pastas...")
    
    # Processar cada split
    for split_name in ['train', 'val', 'test']:
        json_file = os.path.join(processed_path, f'{split_name}.json')
        
        if not os.path.exists(json_file):
            print(f"⚠️  {json_file} não encontrado, pulando...")
            continue
        
        print(f"\n📂 Processando {split_name}...")
        
        # Carregar JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        paths = data['paths']
        labels = data['labels']
        
        print(f"   Total de imagens: {len(paths)}")
        
        # Criar diretórios para cada emoção
        for emotion_idx, emotion_folder in emotion_map.items():
            split_dir = os.path.join(processed_path, split_name, emotion_folder)
            os.makedirs(split_dir, exist_ok=True)
        
        # Copiar imagens para as pastas correspondentes
        for img_path, label in tqdm(zip(paths, labels), total=len(paths), desc=f'  Copiando {split_name}'):
            if not os.path.exists(img_path):
                continue
            
            emotion_folder = emotion_map[label]
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(processed_path, split_name, emotion_folder, img_name)
            
            # Copiar arquivo
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
        
        print(f"   ✅ {split_name} concluído!")
    
    print("\n✅ Conversão completa!")
    
    # Resumo final
    print("\n📊 Estrutura final:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(processed_path, split)
        if os.path.exists(split_path):
            total = 0
            for emotion_folder in emotion_map.values():
                emotion_path = os.path.join(split_path, emotion_folder)
                if os.path.exists(emotion_path):
                    count = len([f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    total += count
            print(f"  {split}: {total} imagens")
    
    print("\n💡 Dica: Você pode deletar os arquivos .json antigos se quiser:")
    print(f"   - {processed_path}/train.json")
    print(f"   - {processed_path}/val.json")
    print(f"   - {processed_path}/test.json")


if __name__ == "__main__":
    convert_json_to_folders()