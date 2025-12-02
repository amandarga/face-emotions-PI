# 🎭 EduFocus - Sistema de Reconhecimento de Emoções

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-68.49%25-brightgreen.svg)

Sistema de análise de emoções faciais em tempo real usando Deep Learning, desenvolvido para monitorar o engajamento de alunos em ambientes educacionais.

[Demo](#-demo) • [Instalação](#-instalação) • [Uso](#-uso) • [Arquitetura](#-arquitetura) • [Resultados](#-resultados)

</div>

---

## 📋 Sobre o Projeto

**EduFocus** é um sistema de Deep Learning desenvolvido para automatizar a análise do engajamento dos alunos em tempo real através da interpretação de expressões faciais. O projeto utiliza visão computacional e processamento de imagens para detectar faces e classificar 7 emoções diferentes com alta precisão.

### Objetivo

Auxiliar professores a medir o engajamento de alunos em aulas remotas ou turmas grandes, fornecendo feedback em tempo real sobre as emoções dos estudantes através da análise de expressões faciais.

### Principais Features

- **Detecção Precisa de Faces**: OpenCV DNN com ResNet-10 SSD
- **Análise de Emoções**: EfficientNet-B2 treinado em FER2013
- **Alta Acurácia**: 68.49% de acurácia no conjunto de validação
- **Múltiplas Faces**: Detecta e analisa várias pessoas simultaneamente
- **Tempo Real**: Processamento frame-by-frame com GPU
- **Sistema de Estabilização**: Reduz flickering nas predições
- **Interface Web**: Aplicação Streamlit interativa e responsiva
- **GPU Acelerado**: Suporte CUDA para processamento rápido

### Emoções Detectadas

| Emoção | Emoji | Cor | Uso Educacional |
|--------|-------|-----|----------------|
| Feliz | 😊 | Verde | Alto engajamento |
| Triste | 😢 | Azul | Possível desinteresse |
| Raiva | 😠 | Vermelho | Frustração |
| Surpresa | 😲 | Amarelo | Descoberta/Interesse |
| Medo | 😨 | Roxo | Ansiedade |
| Nojo | 🤢 | Ciano | Desagrado |
| Neutro | 😐 | Cinza | Atenção passiva |

---

## Demo

### Interface Streamlit

A aplicação web oferece duas funcionalidades principais:

1. **Upload de Imagem**: Analise fotos estáticas
2. **Webcam em Tempo Real**: Detecção contínua via navegador

### Executar Demo Local

```bash
# Interface Web (Streamlit)
streamlit run app.py

# Detecção em Tempo Real (OpenCV)
python src/inference/real_time_improved.py
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- CUDA 11.8+ (opcional, para GPU)
- Webcam (para detecção em tempo real)

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/face-emotions-PI.git
cd face-emotions-PI
```

2. **Crie um ambiente virtual** (recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Download automático de modelos**
Os arquivos necessários (modelo treinado e face detector) são baixados automaticamente na primeira execução do app.

---

## Uso

### 1. Interface Web (Streamlit)

```bash
streamlit run app.py
```

A interface oferece:
- **Upload de imagem**: Arraste e solte fotos para análise
- **Webcam em tempo real**: Acesso direto à câmera pelo navegador
- **Histórico de emoções**: Tabela com todas as detecções
- **Bounding boxes coloridas**: Visualização por emoção

### 2. Detecção em Tempo Real (OpenCV)

```bash
python src/inference/real_time_improved.py
```

Controles:
- **`q` ou `ESC`**: Sair
- **`s`**: Salvar screenshot
- **`r`**: Resetar histórico

### 3. Avaliar Modelo

```bash
python evaluate_model.py
```

Gera relatório completo com:
- Acurácia geral
- Precisão, Recall e F1-Score por emoção
- Matriz de confusão

---

## Arquitetura

### Pipeline de Processamento

```
Imagem/Vídeo 
    ↓
Detecção de Faces (OpenCV DNN)
    ↓
Extração de ROI
    ↓
Preprocessamento (Resize, Normalização)
    ↓
Classificação de Emoção (EfficientNet-B2)
    ↓
Estabilização (Voting System)
    ↓
Visualização (Bounding Box + Label)
```

### Modelos Utilizados

#### 1. Face Detection
- **Modelo**: ResNet-10 SSD (OpenCV DNN)
- **Entrada**: Imagens 300x300
- **Saída**: Coordenadas de faces detectadas
- **Threshold**: 50% de confiança

#### 2. Emotion Recognition
- **Backbone**: EfficientNet-B2 (ImageNet pretrained)
- **Input Size**: 96x96 pixels
- **Classes**: 7 emoções
- **Output**: Probabilidades softmax

### Estrutura do Projeto

```
ProjetoPI/
├── app.py                          # Interface Streamlit
├── evaluate_model.py               # Script de avaliação
├── requirements.txt                # Dependências Python
├── README.md                       # Documentação
├── configs/
│   └── config.yaml                 # Configurações do modelo
├── src/
│   ├── data/
│   │   ├── dataset.py               # Dataset e data augmentation
│   │   └── prepare_dataset.py       # Preparação do FER2013
│   ├── models/
│   │   └── emotion_model.py         # Arquitetura CNN
│   ├── training/
│   │   ├── train_emotions.py       # Treinamento principal
│   │   └── mixup.py                 # Data augmentation (Mixup/CutMix)
│   └── inference/
│       └── real_time_improved.py   # Detecção em tempo real (OpenCV)
├── utils/
│   ├── download_face_detector.py   # Download do detector de faces
│   ├── download_fer2013.py         # Download do dataset
│   └── verify_dataset.py           # Verificação do dataset
├── models/
│   ├── checkpoints/emotions/
│   │   └── best.pth                 # Modelo treinado (97MB)
│   └── face_detector/
│       ├── deploy.prototxt          # Arquitetura do detector
│       └── res10_300x300_ssd_*.caffemodel  # Pesos do detector
└── data/
    ├── raw/emotions/                # Dataset original FER2013
    └── processed/emotions/          # Dataset pré-processado
        ├── train/                   # ~28,000 imagens
        ├── val/                     # ~3,500 imagens
        └── test/                    # ~3,500 imagens
```

---

## Dataset

### FER2013 (Facial Expression Recognition 2013)

- **Fonte**: Kaggle Challenge
- **Tamanho**: ~35,000 imagens
- **Resolução**: 48x48 pixels (grayscale)
- **Divisão**: 80% treino, 10% validação, 10% teste

### Data Augmentation Aplicada

- **Transformações Geométricas**: Rotação, escala, shift
- **Horizontal Flip**: 50% de probabilidade
- **Ruído e Blur**: Gaussian noise, motion blur
- **Distorções**: Grid distortion, elastic transform
- **Cor e Iluminação**: Brightness, contrast, hue, saturation
- **Dropout de Patches**: CoarseDropout
- **Mixup**: Combinação de imagens (20% das batches)
- **Label Smoothing**: ε=0.1

---

## Resultados

### Métricas Gerais

| Métrica | Valor |
|---------|-------|
| **Acurácia de Validação** | 68.49% |
| **Acurácia de Treino** | 57.85% |
| **Épocas Treinadas** | 21/30 |
| **Backbone** | EfficientNet-B2 |

### Performance por Emoção

As emoções são classificadas com diferentes níveis de confiança. Para métricas detalhadas por emoção, execute:
```bash
python evaluate_model.py
```

### Técnicas de Otimização

- **Backbone**: EfficientNet-B2 (pretrained no ImageNet)
- **Input Size**: 96x96 pixels
- **Batch Size**: 24
- **Learning Rate**: 0.0001
- **Regularização**: Dropout (0.4), Weight Decay (0.0001)
- **Loss Function**: Label Smoothing Cross Entropy (ε=0.1)
- **Optimizer**: AdamW com Cosine Annealing Scheduler
- **Data Augmentation**: Mixup (α=0.2, prob=0.5)
- **Early Stopping**: Paciência de 10 épocas
- **Épocas Máximas**: 30

---

## Deploy

### Streamlit Cloud (Recomendado)

1. **Faça push do código para GitHub**
2. **Acesse** [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Conecte seu repositório**
4. **Configure**:
   - Main file: `app.py`
   - Python version: 3.10
5. **Deploy automático!**

O modelo e face detector são baixados automaticamente na primeira execução.

### Requisitos de Hardware

- **Mínimo**: CPU, 2GB RAM
- **Recomendado**: GPU (CUDA), 4GB RAM
- **Cloud**: Funciona em tier gratuito do Streamlit Cloud

---

## Tecnologias Utilizadas

- **Deep Learning**: PyTorch 2.7.1
- **Visão Computacional**: OpenCV 4.x
- **Backbone**: EfficientNet-B2 (timm)
- **Interface**: Streamlit 1.40
- **Data Augmentation**: Albumentations
- **Métricas**: scikit-learn
- **Processamento**: NumPy, Pandas
- **Download**: gdown (Google Drive)

---

## Equipe

**Grupo 1 - EduFocus**
- Projeto Integrador - Senac
- Curso: Deep Learning
- Período: 2025

---

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## Próximos Passos

- [ ] Adicionar detecção de emoções em vídeos gravados
- [ ] Implementar dashboard de analytics
- [ ] Exportar relatórios em PDF

---

<div align="center">

**Desenvolvido com ❤️ para melhorar a educação**

[Reportar Bug](https://github.com/seu-usuario/face-emotions-PI/issues) • [Solicitar Feature](https://github.com/seu-usuario/face-emotions-PI/issues)

</div>