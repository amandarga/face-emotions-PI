# src/inference/real_time_improved.py
import cv2
import torch
import yaml
import numpy as np
import sys
import os
from collections import deque

sys.path.append('src')
from models.emotion_model import EmotionCNN
from data.dataset import get_val_transforms

class ImprovedFaceEmotionDetector:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.emotions = self.config['emotions']['classes']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📦 Carregando modelos...")
        
        # Carregar OpenCV DNN Face Detector
        prototxt = 'models/face_detector/deploy.prototxt'
        caffemodel = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        
        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            print("❌ Modelo de detecção facial não encontrado!")
            print("Execute: python utils/download_face_detector.py")
            sys.exit(1)
        
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print("✅ Detector de faces: CPU (OpenCV DNN)")
        
        # Carregar modelo de emoções
        emotion_path = 'models/checkpoints/emotions/best.pth'
        checkpoint = torch.load(emotion_path, map_location=self.device)
        
        self.emotion_model = EmotionCNN(
            num_classes=len(self.emotions),
            backbone=self.config['emotion_recognition']['backbone'],
            pretrained=False
        )
        self.emotion_model.load_state_dict(checkpoint['model_state_dict'])
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Mostrar acurácia
        val_acc = checkpoint.get('val_acc', 0)
        acc_display = val_acc if val_acc > 1 else val_acc * 100
        print(f"✅ Modelo de emoções: {self.device} (Acc: {acc_display:.2f}%)")
        
        self.transform = get_val_transforms(
            self.config['emotion_recognition']['img_size']
        )
        
        self.emotion_colors = {
            'feliz': (0, 255, 0),
            'triste': (255, 0, 0),
            'raiva': (0, 0, 255),
            'surpresa': (255, 255, 0),
            'neutro': (128, 128, 128),
            'medo': (128, 0, 128),
            'nojo': (0, 128, 128)
        }
        
        # Sistema de estabilização de emoções
        self.emotion_history = {}
        self.stable_emotions = {}
        self.history_size = 7
        self.min_consistency = 5
    
    def detect_faces(self, frame):
        """Detectar faces usando OpenCV DNN"""
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                faces.append((x1, y1, x2, y2, confidence))
        
        return faces
    
    def predict_emotion(self, face_img):
        """Prever emoção de uma face"""
        if face_img.size == 0:
            return 'neutro', 0.0
        
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=face_rgb)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.emotion_model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
            
            emotion = self.emotions[predicted.item()]
            conf = confidence.item()
            return emotion, conf
        except:
            return 'neutro', 0.0
    
    def get_stable_emotion(self, face_id, current_emotion, confidence):
        """Sistema de estabilização: só muda emoção se for consistente"""
        # Ignorar detecções com baixa confiança
        if confidence < 0.4:
            return self.stable_emotions.get(face_id, (current_emotion, confidence))
        
        # Inicializar histórico se não existir
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_size)
            self.stable_emotions[face_id] = (current_emotion, confidence)
        
        # Adicionar emoção atual ao histórico
        self.emotion_history[face_id].append((current_emotion, confidence))
        
        # Se histórico ainda não está cheio, manter emoção atual
        if len(self.emotion_history[face_id]) < self.min_consistency:
            return self.stable_emotions[face_id]
        
        # Contar votos das últimas detecções
        emotion_votes = {}
        total_conf = {}
        
        for emo, conf in self.emotion_history[face_id]:
            emotion_votes[emo] = emotion_votes.get(emo, 0) + 1
            total_conf[emo] = total_conf.get(emo, 0) + conf
        
        # Encontrar emoção mais votada
        winning_emotion = max(emotion_votes, key=emotion_votes.get)
        votes = emotion_votes[winning_emotion]
        
        # Só muda se tiver votos suficientes
        if votes >= self.min_consistency:
            avg_conf = total_conf[winning_emotion] / votes
            self.stable_emotions[face_id] = (winning_emotion, avg_conf)
        
        return self.stable_emotions[face_id]
    
    def run(self, source=0):
        """Executar detecção em tempo real"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ Erro ao abrir câmera")
            return
        
        print("\n🎥 Sistema iniciado!")
        print("📊 Sistema de estabilização ativo")
        print(f"   - Histórico: últimos {self.history_size} frames")
        print(f"   - Mínimo: {self.min_consistency} detecções consistentes para mudar")
        print("\nPressione 'q' ou ESC para sair\n")
        
        frame_count = 0
        emotion_update_interval = 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            faces = self.detect_faces(frame)
            current_face_ids = set()
            
            for x1, y1, x2, y2, conf in faces:
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                face_id = f"{x1//20}_{y1//20}"
                current_face_ids.add(face_id)
                
                # Analisar emoção periodicamente
                if frame_count % emotion_update_interval == 0:
                    raw_emotion, raw_conf = self.predict_emotion(face)
                    emotion, emotion_conf = self.get_stable_emotion(
                        face_id, raw_emotion, raw_conf
                    )
                else:
                    emotion, emotion_conf = self.stable_emotions.get(
                        face_id, ('neutro', 0.0)
                    )
                
                # Desenhar interface
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Box da face
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label com background
                label = f"{emotion} ({emotion_conf*100:.1f}%)"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_h - 10), 
                    (x1 + label_w, y1), 
                    color, 
                    -1
                )
                
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                # Confiança da detecção
                cv2.putText(
                    frame, 
                    f"Det: {conf*100:.1f}%", 
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    1
                )
            
            # Limpar histórico de faces que saíram
            for fid in list(self.emotion_history.keys()):
                if fid not in current_face_ids:
                    del self.emotion_history[fid]
                    if fid in self.stable_emotions:
                        del self.stable_emotions[fid]
            
            # Info no canto
            cv2.putText(
                frame, 
                f"Faces: {len(faces)}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow('Detecção Facial + Emoções', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Sistema encerrado!")

if __name__ == "__main__":
    detector = ImprovedFaceEmotionDetector()
    detector.run()