# utils/verify_dataset.py
import os
from collections import Counter

def verify_dataset():
    """Verificar se o dataset foi instalado corretamente"""
    
    print("🔍 Verificando dataset...\n")
    
    emotions_dir = 'data/raw/emotions'
    
    if not os.path.exists(emotions_dir):
        print(f"❌ Pasta não encontrada: {emotions_dir}")
        return False
    
    emotions = ['feliz', 'triste', 'raiva', 'surpresa', 'neutro', 'medo', 'nojo']
    
    stats = {}
    total = 0
    
    for emotion in emotions:
        emotion_path = os.path.join(emotions_dir, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"❌ Pasta não encontrada: {emotion}")
            stats[emotion] = 0
            continue
        
        # Contar imagens
        images = [f for f in os.listdir(emotion_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        num_images = len(images)
        stats[emotion] = num_images
        total += num_images
        
        status = "✅" if num_images > 0 else "⚠️ "
        print(f"{status} {emotion}: {num_images} imagens")
    
    print(f"\n{'='*40}")
    print(f"TOTAL: {total} imagens")
    print(f"{'='*40}\n")
    
    # Verificar se está balanceado
    if total > 0:
        min_images = min(stats.values())
        max_images = max(stats.values())
        ratio = max_images / min_images if min_images > 0 else float('inf')
        
        if ratio < 3:
            print("✅ Dataset bem balanceado")
        else:
            print("⚠️  Dataset desbalanceado (considere data augmentation)")
        
        # Recomendações
        print("\n📋 Recomendações:")
        if total < 1000:
            print("   ⚠️  Dataset pequeno - considere usar data augmentation")
        elif total < 5000:
            print("   ✅ Dataset adequado para treinamento básico")
        else:
            print("   ✅ Dataset bom para treinamento robusto")
        
        return True
    else:
        print("❌ Nenhuma imagem encontrada")
        return False

if __name__ == "__main__":
    verify_dataset()