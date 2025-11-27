# utils/download_fer2013.py
import os
import shutil
import zipfile
from pathlib import Path

def download_fer2013_kaggle():
    """
    Baixar FER2013 via Kaggle API
    
    Pré-requisitos:
    1. pip install kaggle
    2. Conta no Kaggle
    3. API token em ~/.kaggle/kaggle.json
    """
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("📥 Iniciando download do FER2013 via Kaggle...\n")
        
        # Autenticar
        api = KaggleApi()
        api.authenticate()
        print("✅ Autenticado no Kaggle")
        
        # Criar diretório temporário
        temp_dir = 'data/temp_fer2013'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download
        print("📦 Baixando dataset (pode demorar alguns minutos)...")
        api.dataset_download_files(
            'msambare/fer2013',
            path=temp_dir,
            unzip=True
        )
        
        print("✅ Download concluído!\n")
        
        # Organizar
        organize_fer2013(temp_dir)
        
        # Limpar
        shutil.rmtree(temp_dir)
        print("\n✅ Dataset FER2013 instalado com sucesso!")
        
        return True
        
    except ImportError:
        print("❌ Kaggle não instalado. Execute: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Tente o download manual (opção 2)")
        return False

def organize_fer2013(source_dir):
    """Organizar FER2013 na estrutura correta"""
    
    print("📂 Organizando arquivos...\n")
    
    # Mapeamento de emoções (inglês → português)
    emotions_map = {
        'angry': 'raiva',
        'disgust': 'nojo',
        'fear': 'medo',
        'happy': 'feliz',
        'sad': 'triste',
        'surprise': 'surpresa',
        'neutral': 'neutro'
    }
    
    target_dir = 'data/raw/emotions'
    os.makedirs(target_dir, exist_ok=True)
    
    # Processar train e test
    for split in ['train', 'test']:
        split_path = os.path.join(source_dir, split)
        
        if not os.path.exists(split_path):
            continue
        
        for en_emotion, pt_emotion in emotions_map.items():
            src = os.path.join(split_path, en_emotion)
            dst = os.path.join(target_dir, pt_emotion)
            
            if not os.path.exists(src):
                continue
            
            # Criar pasta destino
            os.makedirs(dst, exist_ok=True)
            
            # Copiar imagens
            count = 0
            for img_file in os.listdir(src):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(src, img_file)
                    # Renomear para evitar conflitos train/test
                    dst_file = os.path.join(dst, f"{split}_{img_file}")
                    shutil.copy2(src_file, dst_file)
                    count += 1
            
            print(f"✅ {pt_emotion}: +{count} imagens ({split})")
    
    # Mostrar estatísticas finais
    print("\n📊 Estatísticas finais:")
    total = 0
    for emotion in emotions_map.values():
        emotion_path = os.path.join(target_dir, emotion)
        if os.path.exists(emotion_path):
            num_images = len([f for f in os.listdir(emotion_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   {emotion}: {num_images} imagens")
            total += num_images
    
    print(f"\n   TOTAL: {total} imagens")

def download_manual_instructions():
    """Instruções para download manual"""
    
    print("\n" + "="*60)
    print("📖 INSTRUÇÕES PARA DOWNLOAD MANUAL")
    print("="*60)
    print("\n1️⃣  Acesse: https://www.kaggle.com/datasets/msambare/fer2013")
    print("\n2️⃣  Clique em 'Download' (precisa estar logado)")
    print("\n3️⃣  Extraia o arquivo fer2013.zip")
    print("\n4️⃣  Execute:")
    print("     python utils/organize_manual_fer2013.py <caminho_extraido>")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("="*60)
    print("     DOWNLOAD FER2013 DATASET")
    print("="*60 + "\n")
    
    print("Escolha uma opção:")
    print("1 - Download automático via Kaggle API (recomendado)")
    print("2 - Instruções para download manual")
    
    choice = input("\nOpção (1 ou 2): ").strip()
    
    if choice == "1":
        success = download_fer2013_kaggle()
        if not success:
            download_manual_instructions()
    elif choice == "2":
        download_manual_instructions()
    else:
        print("❌ Opção inválida")