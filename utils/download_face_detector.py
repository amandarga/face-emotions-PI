# utils/download_face_detector.py
import os
import urllib.request

def download_face_detector():
    """Download OpenCV DNN face detector model"""
    
    model_dir = 'models/face_detector'
    os.makedirs(model_dir, exist_ok=True)
    
    files = {
        'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    }
    
    print("📥 Baixando modelo de detecção facial...")
    
    for filename, url in files.items():
        filepath = os.path.join(model_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {filename} já existe")
            continue
        
        print(f"⏳ Baixando {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ {filename} baixado!")
        except Exception as e:
            print(f"❌ Erro ao baixar {filename}: {e}")
            return False
    
    print("\n✅ Modelo de detecção facial pronto!")
    return True

if __name__ == "__main__":
    download_face_detector()