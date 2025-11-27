# 🎭 Face Emotions Detection - Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-68.38%25-brightgreen.svg)

Sistema de detecção facial e análise de emoções em tempo real usando Deep Learning.

[Demo](#-demo) • [Instalação](#-instalação) • [Uso](#-uso) • [Resultados](#-resultados)

</div>

---

## 📋 Sobre o Projeto

Sistema profissional de análise de emoções faciais em tempo real utilizando Deep Learning. O projeto detecta faces em vídeo e classifica 7 emoções diferentes com alta precisão.

### ✨ Principais Features

- 🎯 **Detecção Precisa de Faces**: OpenCV DNN com ResNet SSD
- 🧠 **Análise de Emoções**: EfficientNet-B0 treinado em FER2013
- 📊 **Alta Acurácia**: 68.38% de acurácia (Top 20% no FER2013)
- 👥 **Múltiplas Faces**: Detecta e analisa várias pessoas simultaneamente
- 🎬 **Tempo Real**: ~30-60 FPS com GPU
- 🔄 **Sistema de Estabilização**: Smoothing para reduzir variações rápidas
- ⚡ **GPU Acelerado**: Suporte CUDA para processamento rápido

### 🎭 Emoções Detectadas

| Emoção | Emoji | Cor |
|--------|-------|-----|
| Feliz (Happy) | 😊 | Verde |
| Triste (Sad) | 😢 | Azul |
| Raiva (Angry) | 😠 | Vermelho |
| Surpresa (Surprise) | 😲 | Amarelo |
| Medo (Fear) | 😨 | Roxo |
| Nojo (Disgust) | 🤢 | Ciano |
| Neutro (Neutral) | 😐 | Cinza |

---

## 🎥 Demo

> 📸 *Adicione aqui um GIF ou vídeo mostrando o sistema funcionando*

```bash
# Para testar o sistema:
python src/inference/real_time_improved.py