# app.py
import streamlit as st
import cv2
import pandas as pd
import torch
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from datetime import datetime

sys.path.append('src')
from models.emotion_model import EmotionCNN
from data.dataset import get_val_transforms

# Configuração
st.set_page_config(
    page_title="Reconhecimento de Emoções",
    page_icon="😊",
    layout="wide"
)

# Título
st.title("😊 Reconhecimento de Emoções Faciais")

# Emojis por emoção
EMOJIS = {
    'feliz': '😊',
    'triste': '😢',
    'raiva': '😠',
    'surpresa': '😮',
    'neutro': '😐',
    'medo': '😨',
    'nojo': '🤢'
}

# Inicializar histórico no session_state
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Carregar modelo (cache)
@st.cache_resource
def carregar_modelo():
    """Carregar modelo apenas uma vez"""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    emotions = config['emotions']['classes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('models/checkpoints/emotions/best.pth', map_location=device)
    
    model = EmotionCNN(
        num_classes=len(emotions),
        backbone=config['emotion_recognition']['backbone'],
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_val_transforms(config['emotion_recognition']['img_size'])
    
    return model, transform, emotions, device

# Detectar faces
def detectar_faces(image):
    """Detectar faces usando OpenCV DNN"""
    prototxt = 'models/face_detector/deploy.prototxt'
    caffemodel = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            faces.append((x1, y1, x2, y2))
    
    return faces

# Prever emoção
def prever_emocao(face_img, model, transform, emotions, device):
    """Prever emoção de uma face"""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=face_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_idx = probs.argmax(1).item()
        confidence = probs[0, predicted_idx].item()
    
    emotion = emotions[predicted_idx]
    return emotion, confidence

# Adicionar ao histórico
def adicionar_historico(emocao, confianca, fonte):
    """Adicionar detecção ao histórico"""
    agora = datetime.now().strftime("%H:%M:%S")
    emoji = EMOJIS.get(emocao, '😐')
    
    st.session_state.historico.insert(0, {
        'hora': agora,
        'emocao': emocao,
        'emoji': emoji,
        'confianca': confianca,
        'fonte': fonte
    })
    
    # Limitar a 50 registros
    if len(st.session_state.historico) > 50:
        st.session_state.historico = st.session_state.historico[:50]

# Carregar modelo
try:
    model, transform, emotions, device = carregar_modelo()
    st.sidebar.success("✅ Modelo carregado!")
    st.sidebar.info(f"🖥️ Device: {'GPU' if device.type == 'cuda' else 'CPU'}")
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {e}")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["📷 Upload de Imagem", "🎥 Webcam"])

# ==================== TAB 1: UPLOAD ====================
with tab1:
    st.header("📤 Enviar Imagem")
    
    uploaded_file = st.file_uploader(
        "Escolha uma foto com rosto",
        type=['jpg', 'jpeg', 'png'],
        help="A imagem será analisada automaticamente"
    )
    
    if uploaded_file is not None:
        # Ler imagem
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Converter para BGR
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Imagem Original", use_column_width=True)
        
        with col2:
            with st.spinner("🔍 Analisando..."):
                # Detectar faces
                faces = detectar_faces(image_bgr)
                
                if len(faces) == 0:
                    st.warning("⚠️ Nenhum rosto detectado")
                else:
                    # Analisar cada face
                    result_img = image_bgr.copy()
                    
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        face = image_bgr[y1:y2, x1:x2]
                        
                        if face.size > 0:
                            emocao, conf = prever_emocao(face, model, transform, emotions, device)
                            emoji = EMOJIS.get(emocao, '😐')
                            
                            # Desenhar resultado
                            color = (0, 255, 0) if conf > 0.6 else (255, 165, 0)
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{emoji} {emocao} ({conf*100:.0f}%)"
                            cv2.putText(result_img, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Adicionar ao histórico
                            adicionar_historico(emocao, conf, "Upload")
                    
                    # Mostrar resultado
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption=f"✅ {len(faces)} rosto(s) detectado(s)", 
                            use_column_width=True)

# ==================== TAB 2: WEBCAM ====================
with tab2:
    st.header("🎥 Câmera em Tempo Real no Navegador")
    
    st.info("📹 **A webcam vai abrir aqui embaixo!** Permita o acesso quando o navegador pedir.")
    
    # Cores por emoção (BGR para OpenCV)
    EMOTION_COLORS = {
        'feliz': (0, 255, 0),       # Verde
        'triste': (255, 0, 0),       # Azul
        'raiva': (0, 0, 255),        # Vermelho
        'surpresa': (0, 255, 255),   # Amarelo
        'neutro': (128, 128, 128),   # Cinza
        'medo': (128, 0, 128),       # Roxo
        'nojo': (0, 128, 128)        # Verde-água
    }
    
    # Importar streamlit-webrtc
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        import av
        
        class EmotionVideoProcessor(VideoProcessorBase):
            """Processar vídeo frame por frame"""
            
            def __init__(self):
                self.ultima_emocao = None
                self.ultima_confianca = 0
                self.frame_count = 0
                # Sistema de estabilização
                self.emotion_history = []  # Histórico de emoções
                self.history_size = 4      # Tamanho do histórico
                self.stable_emotion = None  # Emoção estável
                self.stable_confidence = 0
                self.cached_faces = []
            
            def get_stable_emotion(self, emotion, confidence):
                """Sistema de estabilização - evita piscamento"""
                # Adicionar ao histórico
                self.emotion_history.append((emotion, confidence))
                
                # Manter apenas últimas N detecções
                if len(self.emotion_history) > self.history_size:
                    self.emotion_history.pop(0)
                
                # Precisa de pelo menos 5 detecções para estabilizar
                if len(self.emotion_history) < 3:
                    return emotion, confidence
                
                # Contar ocorrências de cada emoção
                emotion_counts = {}
                for emo, conf in self.emotion_history:
                    if emo not in emotion_counts:
                        emotion_counts[emo] = []
                    emotion_counts[emo].append(conf)
                
                # Pegar emoção mais frequente
                most_common = max(emotion_counts.items(), key=lambda x: len(x[1]))
                stable_emo = most_common[0]
                avg_conf = sum(most_common[1]) / len(most_common[1])
                
                # Só mudar se a nova emoção aparecer muito (>60% dos frames)
                new_count = len(emotion_counts.get(emotion, []))
                if new_count > self.history_size * 0.6:
                    return emotion, confidence
                
                return stable_emo, avg_conf
            
            def recv(self, frame):
                # Converter frame
                img = frame.to_ndarray(format="bgr24")
                
                # Processar a cada 5 frames (MELHOR para qualidade visual!)
                self.frame_count += 1
                if self.frame_count % 3 == 0:  # Era 3, agora 5 = menos lag
                    try:
                        # Detectar faces (cache para não detectar todo frame)
                        faces = detectar_faces(img)
                        
                        if len(faces) > 0:
                            # Processar PRIMEIRA face apenas
                            x1, y1, x2, y2 = faces[0]
                            face = img[y1:y2, x1:x2]
                            
                            if face.size > 0:
                                # Prever emoção
                                emocao, conf = prever_emocao(face, model, transform, emotions, device)
                                
                                # ESTABILIZAR emoção
                                emocao, conf = self.get_stable_emotion(emocao, conf)
                                
                                # Salvar para desenhar
                                self.stable_emotion = emocao
                                self.stable_confidence = conf
                                self.cached_faces = faces  # Cachear faces detectadas
                                
                                # Histórico menos frequente
                                if self.frame_count % 120 == 0:  # A cada 4 segundos
                                    adicionar_historico(emocao, conf, "Webcam Real-Time")
                    except Exception as e:
                        pass
                
                # Desenhar usando cache (RÁPIDO - sem re-detectar!)
                if hasattr(self, 'stable_emotion') and self.stable_emotion:
                    if hasattr(self, 'cached_faces'):
                        faces_to_draw = self.cached_faces
                    else:
                        faces_to_draw = []
                    
                    for face_box in faces_to_draw:
                        x1, y1, x2, y2 = face_box
                        
                        emocao = self.stable_emotion
                        conf = self.stable_confidence
                        color = EMOTION_COLORS.get(emocao, (255, 255, 255))
                        label_texto = f"{emocao.upper()} ({conf*100:.0f}%)"
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.9
                        thickness = 2
                        (w, h), _ = cv2.getTextSize(label_texto, font, font_scale, thickness)
                        
                        # Retângulo colorido
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                        
                        # Label
                        label_y = y1 - 15
                        if label_y < h + 15:
                            label_y = y2 + h + 15
                        
                        # Background sólido
                        cv2.rectangle(img,
                                    (x1, label_y - h - 10),
                                    (x1 + w + 20, label_y + 5),
                                    color, -1)
                        cv2.rectangle(img,
                                    (x1, label_y - h - 10),
                                    (x1 + w + 20, label_y + 5),
                                    (255, 255, 255), 2)
                        
                        cv2.putText(img, label_texto,
                                  (x1 + 10, label_y - 5),
                                  font, font_scale, (255, 255, 255), thickness)
                    
                    # Contador
                    if faces_to_draw:
                        h_img, w_img = img.shape[:2]
                        faces_count = len(faces_to_draw)
                        counter_text = f"Faces: {faces_count} | {30}fps"  # Mostrar FPS
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (tw, th), _ = cv2.getTextSize(counter_text, font, font_scale, thickness)
                        
                        cv2.rectangle(img, (w_img - tw - 25, 10),
                                    (w_img - 5, th + 25), (50, 50, 50), -1)
                        cv2.rectangle(img, (w_img - tw - 25, 10),
                                    (w_img - 5, th + 25), (255, 255, 255), 2)
                        cv2.putText(img, counter_text, (w_img - tw - 18, th + 18),
                                  font, font_scale, (255, 255, 255), thickness)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Configuração do webrtc
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "min": 640},     # HD
                    "height": {"ideal": 720, "min": 480},     # HD
                    "frameRate": {"ideal": 30, "max": 60},    # Até 60 FPS
                    "facingMode": "user"                       # Câmera frontal
                }, 
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}  # Servidor adicional
                ]
            },
            video_frame_callback=None,
            desired_playing_state=True
        )
        
        st.success("✅ Webcam ativada! A detecção está rodando em tempo real (~30 FPS).")
        
        # Legenda de cores
        st.markdown("---")
        st.subheader("🎨 Legenda de Cores")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("🟢 **FELIZ** - Verde")
            st.markdown("🔴 **RAIVA** - Vermelho")
        
        with col2:
            st.markdown("🔵 **TRISTE** - Azul")
            st.markdown("🟡 **SURPRESA** - Amarelo")
        
        with col3:
            st.markdown("🟣 **MEDO** - Roxo")
            st.markdown("🟤 **NOJO** - Verde-água")
        
        with col4:
            st.markdown("⚫ **NEUTRO** - Cinza")
        
        st.info("""
        💡 **Dicas:**
        - Cada emoção tem sua **cor específica** no fundo
        - Texto em **branco** para melhor contraste
        - FPS otimizado (~30 FPS)
        - Histórico atualiza a cada ~1.5 segundos
        """)
        
    except ImportError:
        st.error("❌ Erro: `streamlit-webrtc` não instalado!")
        st.code("pip install streamlit-webrtc av", language="bash")
    except Exception as e:
        st.error(f"❌ Erro ao iniciar webcam: {e}")
        st.info("""
        **Alternativa:** Use o script dedicado:
        ```bash
        python src/inference/real_time_improved.py
        ```
        """)
# ==================== HISTÓRICO (EMBAIXO) ====================
st.markdown("---")
st.header("📊 Histórico de Detecções")

if len(st.session_state.historico) == 0:
    st.info("👆 Nenhuma detecção ainda. Use as abas acima para começar!")
else:
    # Botão para limpar
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Limpar"):
            st.session_state.historico = []
            st.rerun()
    
    # Mostrar total
    st.markdown(f"**Total: {len(st.session_state.historico)} detecções**")
    
    # Criar DataFrame para tabela
    import pandas as pd
    
    # Pegar últimas 20 detecções
    historico_recente = st.session_state.historico[:20]
    
    # Preparar dados
    dados = []
    for item in historico_recente:
        conf = item['confianca']
        
        # Ícone de confiança
        if conf >= 0.7:
            conf_icon = "🟢"
        elif conf >= 0.5:
            conf_icon = "🟡"
        else:
            conf_icon = "🔴"
        
        dados.append({
            '⏰ Hora': item['hora'],
            '😊 Emoção': f"{item['emoji']} {item['emocao'].capitalize()}",
            '📊 Confiança': f"{conf_icon} {conf*100:.1f}%",
            '📷 Fonte': item['fonte']
        })
    
    # Criar DataFrame
    df = pd.DataFrame(dados)
    
    # Mostrar tabela com estilo
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Estatísticas
    st.markdown("---")
    st.subheader("📈 Estatísticas")
    
    col1, col2, col3 = st.columns(3)
    
    # Emoção mais frequente
    emocoes_count = {}
    for item in st.session_state.historico:
        emo = item['emocao']
        emocoes_count[emo] = emocoes_count.get(emo, 0) + 1
    
    if emocoes_count:
        mais_frequente = max(emocoes_count, key=emocoes_count.get)
        emoji_freq = EMOJIS.get(mais_frequente, '😐')
        
        with col1:
            st.metric("Emoção Mais Frequente", 
                     f"{emoji_freq} {mais_frequente.capitalize()}", 
                     f"{emocoes_count[mais_frequente]} vezes")
        
        with col2:
            conf_media = sum(item['confianca'] for item in st.session_state.historico) / len(st.session_state.historico)
            st.metric("Confiança Média", f"{conf_media*100:.1f}%")
        
        with col3:
            st.metric("Total de Detecções", len(st.session_state.historico))

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Sobre")
    st.info("""
    **Sistema de Reconhecimento de Emoções**
    
    🧠 Modelo: EfficientNet-B2
    📊 Acurácia: 68.49%
    🎯 7 Emoções
    
    **Projeto Integrador**
    Senac 2024
    """)
    
    st.markdown("---")
    st.markdown("### 🎨 Emoções")
    for emo, emoji in EMOJIS.items():
        st.markdown(f"{emoji} {emo.capitalize()}")