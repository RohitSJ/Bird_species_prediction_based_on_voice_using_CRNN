# Bird Species Prediction from Voice and Images 🐦🔊📷

A comprehensive deep learning system that classifies bird species using both audio recordings (CRNN model) and images (VGG16 model), featuring a Flask web interface for seamless user interaction and multi-modal prediction capabilities.

---

## 🎬 Live Demo Videos

### 🎙️ Audio Classification Demo
<div align="center">
  <img src="demos/model1.gif" width="700" alt="Audio Classification Demo">
  <p><em>Upload bird audio → AI processes sound → Instant species identification with confidence scores</em></p>
</div>

### 🖼️ Image Classification Demo  
<div align="center">
  <img src="demos/model2.gif" width="700" alt="Image Classification Demo">
  <p><em>Upload bird image → AI analyzes features → Species prediction with scientific information</em></p>
</div>

---

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Key Features

- **🎯 Multi-Modal AI**: Dual classification system using both audio and visual data
- **🎵 Advanced Audio Processing**: CRNN with BiLSTM + Attention mechanism
- **🖼️ Computer Vision**: Fine-tuned VGG16 for image recognition
- **🌐 Web Interface**: User-friendly Flask application
- **📚 Rich Database**: Scientific classification + Wikipedia integration
- **🎼 Multi-Format Support**: `.wav`, `.mp3`, `.flac`, `.webm` audio files
- **📊 Confidence Scoring**: Detailed prediction probabilities
- **🔍 Top-5 Results**: Multiple prediction candidates
- **📱 Responsive Design**: Works on desktop and mobile

---

## 🎯 Quick Start

### 1️⃣ Clone & Setup
\`\`\`bash
git clone https://github.com/yourusername/bird-species-prediction.git
cd bird-species-prediction
pip install -r requirements.txt
\`\`\`

### 2️⃣ Run the Application
\`\`\`bash
python app.py
\`\`\`

### 3️⃣ Open in Browser
\`\`\`
http://localhost:5000
\`\`\`

**That's it! Start uploading bird audio files or images to see the AI in action! 🚀**

---

## 📁 Project Structure

\`\`\`
bird-species-prediction/
├── 🎬 demos/                      # Demo videos and GIFs
│   ├── audio_demo.gif             # Audio classification demo
│   └── image_demo.gif             # Image classification demo
├── 📄 app.py                      # Flask web application
├── 🎵 Audio_model_training.py     # CRNN audio model training
├── 🖼️ Image_model_training.py     # VGG16 image model training
├── 📋 BirdInfo.json               # Bird species database
├── 📦 models/
│   ├── best_bird_model.h5         # Trained CRNN model
│   ├── bird_classifier.h5         # Trained VGG16 model
│   └── class_indices.pkl          # Class mappings
├── 🎨 static/
│   ├── uploads/                   # User uploads
│   └── training_history.png       # Training plots
├── 🌐 templates/
│   ├── home.html                  # Landing page
│   ├── voice.html                 # Audio interface
│   └── image.html                 # Image interface
└── 📋 requirements.txt            # Dependencies
\`\`\`

---

## 🎯 How It Works

### 🔊 Audio Classification Pipeline
\`\`\`
Bird Audio → Mel-Spectrogram → CRNN Model → Species Prediction
     ↓              ↓              ↓              ↓
  .wav/.mp3    Feature Maps    BiLSTM+Attention   Confidence Score
\`\`\`

### 🖼️ Image Classification Pipeline  
\`\`\`
Bird Image → Preprocessing → VGG16 Model → Species Prediction
     ↓            ↓             ↓             ↓
  .jpg/.png   224x224 RGB   Transfer Learning  Top-5 Results
\`\`\`

---

## 📊 Model Performance

<div align="center">

| 🎯 Model | 🏗️ Architecture | 📈 Accuracy | ⚡ Speed |
|----------|----------------|-------------|---------|
| **Audio** | CRNN (BiLSTM + Attention) | **90%** | <1s |
| **Image** | VGG16 (Transfer Learning) | **85%** | <1s |

</div>

### 📈 Training Results
<div align="center">
  <img src="static/training_history.png" width="600" alt="Training History">
</div>

---

## 🗃️ Dataset & Species Coverage

- **🐦 Total Species**: 143 bird species
- **🎵 Audio Samples**: 10,000+ recordings  
- **🖼️ Image Samples**: 8,000+ high-quality images
- **🌍 Coverage**: Songbirds, Raptors, Waterbirds, Game Birds, Exotic Species

---

## 🔗 API Usage

### Audio Prediction
\`\`\`bash
curl -X POST -F "file=@bird_sound.wav" http://localhost:5000/predict_voice
\`\`\`

### Image Prediction
\`\`\`bash
curl -X POST -F "file=@bird_image.jpg" http://localhost:5000/predict_image
\`\`\`

### Response Format
\`\`\`json
{
  "prediction": "Indian Robin",
  "confidence": 0.92,
  "scientific_name": "Copsychus fulicatus",
  "wiki_summary": "The Indian robin is a species of bird...",
  "wiki_image": "https://upload.wikimedia.org/...jpg",
  "top_predictions": [
    {"species": "Indian Robin", "confidence": 0.92},
    {"species": "Oriental Magpie Robin", "confidence": 0.05}
  ]
}
\`\`\`

---

## 🧠 Model Training

### Train Audio Model (CRNN)
\`\`\`bash
python Audio_model_training.py
\`\`\`

### Train Image Model (VGG16)  
\`\`\`bash
python Image_model_training.py
\`\`\`

---

## ⚙️ Technical Architecture

### 🎵 Audio Model (CRNN)
- **Input**: Mel-spectrogram (500, 33)
- **CNN Layers**: Spatial feature extraction
- **BiLSTM**: Temporal pattern recognition  
- **Attention**: Focus on important segments
- **Output**: 143 species classification

### 🖼️ Image Model (VGG16)
- **Input**: RGB images (224, 224, 3)
- **Base**: Pre-trained VGG16 (ImageNet)
- **Fine-tuning**: Bird-specific features
- **Output**: Multi-class classification

---

## 🔧 Configuration

### Audio Processing
```python
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
