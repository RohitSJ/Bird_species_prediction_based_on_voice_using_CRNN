# Bird Species Prediction from Voice and Images 🐦🔊📷

A comprehensive deep learning system that classifies bird species using both audio recordings (CRNN model) and images (VGG16 model), featuring a Flask web interface for seamless user interaction and multi-modal prediction capabilities.

![Project Banner](static/banner.png) <!-- Add your project banner image -->

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Features

- **🎯 Multi-Modal Classification**: Predicts bird species from both **audio recordings** and **images**
- **🎵 CRNN Audio Model**: Convolutional Recurrent Neural Network with BiLSTM and Attention for bird sound classification
- **🖼️ VGG16 Image Model**: Fine-tuned VGG16 transfer learning model for bird image classification
- **🌐 Flask Web Interface**: User-friendly web application with intuitive design
- **📚 BirdInfo Database**: Comprehensive database with scientific classification and Wikipedia integration
- **🎼 Multi-Format Audio Support**: Accepts `.wav`, `.mp3`, `.flac`, `.webm` audio formats
- **📊 Confidence Scoring**: Provides prediction confidence levels for both modalities
- **🔍 Top-5 Predictions**: Shows multiple prediction candidates with confidence scores
- **📱 Responsive Design**: Works seamlessly across desktop and mobile devices

---

## 📁 Project Structure

\`\`\`
bird-species-prediction/
├── 📄 app.py                      # Flask web application
├── 🎵 Audio_model_training.py     # CRNN audio model training script
├── 🖼️ Image_model_training.py     # VGG16 image model training script
├── 📋 BirdInfo.json               # Bird species database with scientific info
├── 📦 models/
│   ├── best_bird_model.h5         # Trained CRNN model for audio
│   ├── bird_classifier.h5         # Trained VGG16 model for images
│   └── class_indices.pkl          # Class label mapping dictionary
├── 🎨 static/
│   ├── uploads/                   # User uploaded files storage
│   ├── training_history.png       # Model training accuracy/loss plots
│   └── banner.png                 # Project banner image
├── 🌐 templates/
│   ├── home.html                   # Landing page template
|   |-- base.html
│   ├── voice.html                 # Audio prediction interface
│   └── image.html                 # Image prediction interface
├── 🎬 demos/                      # Demo videos and GIFs
│   ├── audio_demo.mp4             # Audio classification demo video
│   ├── image_demo.mp4             # Image classification demo video
│   ├── audio_demo.gif             # Audio demo as GIF (optional)
│   └── image_demo.gif             # Image demo as GIF (optional)
├── 📋 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
└── 📜 LICENSE                     # MIT License file
\`\`\`

---

## 📽 Demo Videos

### Option 1: Upload to GitHub and embed (Recommended)

\`\`\`markdown
### 🎙 Audio Classification Demo
![Audio Demo](demos/audio_demo.gif)
*Screen recording showing audio file upload and bird species prediction*

### 🖼 Image Classification Demo  
![Image Demo](demos/image_demo.gif)
*Screen recording showing image upload and classification results*
\`\`\`

### Option 2: Convert to GIF and embed directly

\`\`\`markdown
### 🎙 Audio Classification Demo
<img src="demos/audio_classification_demo.gif" width="600" alt="Audio Classification Demo">

### 🖼 Image Classification Demo
<img src="demos/image_classification_demo.gif" width="600" alt="Image Classification Demo">
\`\`\`

### Option 3: Link to video files in repository

\`\`\`markdown
### 📹 Demo Videos

#### 🎙 Audio Classification Demo
[![Audio Demo](https://img.shields.io/badge/▶️-Watch%20Audio%20Demo-blue?style=for-the-badge)](demos/audio_demo.mp4)

#### 🖼 Image Classification Demo  
[![Image Demo](https://img.shields.io/badge/▶️-Watch%20Image%20Demo-green?style=for-the-badge)](demos/image_demo.mp4)

> 📁 **Note**: Click the badges above to download and view the demo videos
\`\`\`

### Option 4: Upload to cloud storage and embed

\`\`\`markdown
### 🎥 Live Demos

#### 🎙 Audio Classification Walkthrough
<video width="600" controls>
  <source src="https://your-cloud-storage.com/audio_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 🖼 Image Classification Walkthrough
<video width="600" controls>
  <source src="https://your-cloud-storage.com/image_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
\`\`\`

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU support optional but recommended for training

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/yourusername/bird-species-prediction.git
cd bird-species-prediction
\`\`\`

### 2. Create Virtual Environment (Recommended)
\`\`\`bash
# Create virtual environment
python -m venv bird_env

# Activate virtual environment
# On Windows:
bird_env\Scripts\activate
# On macOS/Linux:
source bird_env/bin/activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Download Pre-trained Models
Place the following files in the `models/` directory:
- `best_bird_model.h5` (CRNN audio model)
- `bird_classifier.h5` (VGG16 image model)  
- `class_indices.pkl` (class mappings)

> 📧 **Contact**: If models are not available, email [your-email@example.com] to obtain them.

### 5. Verify Installation
\`\`\`bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
\`\`\`

---

## ▶️ Usage

### 🌐 Web Application

1. **Start the Flask server**:
\`\`\`bash
python app.py
\`\`\`

2. **Open your browser** and navigate to:
\`\`\`
http://localhost:5000
\`\`\`

3. **Use the interface**:
   - **Home Page**: Overview and navigation
   - **Voice Prediction**: Upload audio files for bird species identification
   - **Image Prediction**: Upload images for bird species classification

### 🎵 Audio Prediction
- Supported formats: `.wav`, `.mp3`, `.flac`, `.webm`
- Recommended: Clear recordings with minimal background noise
- Duration: 3-30 seconds optimal

### 🖼️ Image Prediction  
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Recommended: High-quality images with clear bird visibility
- Resolution: Automatically resized to 224x224 pixels

---

## 🧠 Model Training

### 🎵 Train Audio Model (CRNN)
\`\`\`bash
python Audio_model_training.py
\`\`\`

**Audio Model Architecture**:
- **Input**: Mel-spectrogram features (500, 33)
- **CNN Layers**: Feature extraction from spectrograms
- **BiLSTM**: Bidirectional temporal modeling
- **Attention**: Focus on important audio segments
- **Output**: 143 bird species classification

### 🖼️ Train Image Model (VGG16)
\`\`\`bash
python Image_model_training.py
\`\`\`

**Image Model Architecture**:
- **Base Model**: Pre-trained VGG16 (ImageNet weights)
- **Input**: RGB images (224, 224, 3)
- **Fine-tuning**: Last few layers retrained
- **Output**: 143 bird species classification

---

## 🔗 API Endpoints (Optional)

For programmatic access, the application provides REST API endpoints:

### Audio Prediction API
\`\`\`bash
curl -X POST -F "file=@bird_sound.wav" http://localhost:5000/predict_voice
\`\`\`

### Image Prediction API
\`\`\`bash
curl -X POST -F "file=@bird_image.jpg" http://localhost:5000/predict_image
\`\`\`

### Response Format
\`\`\`json
{
  "prediction": "Indian Robin",
  "confidence": 0.92,
  "scientific_name": "Copsychus fulicatus",
  "wiki_summary": "The Indian robin is a species of bird in the family Muscicapidae...",
  "wiki_image": "https://upload.wikimedia.org/...jpg",
  "top_predictions": [
    {"species": "Indian Robin", "confidence": 0.92},
    {"species": "Oriental Magpie Robin", "confidence": 0.05},
    {"species": "White-rumped Shama", "confidence": 0.02}
  ]
}
\`\`\`

---

## ⚙️ Technical Architecture

### 🔊 Audio Processing Pipeline

1. **Audio Loading**: Load audio files using librosa
2. **Preprocessing**: 
   - Normalize audio amplitude
   - Remove silence segments
   - Resample to consistent sample rate
3. **Feature Extraction**:
   - Mel-spectrogram generation
   - MFCC (Mel-Frequency Cepstral Coefficients)
   - Chroma features
   - Spectral contrast
4. **Model Input**: Convert to (500, 33) feature matrix
5. **CRNN Processing**: 
   - CNN layers extract spatial features
   - BiLSTM captures temporal dependencies
   - Attention mechanism focuses on important segments
6. **Classification**: Softmax output for 143 species

### 🖼 Image Processing Pipeline

1. **Image Loading**: Load and validate image files
2. **Preprocessing**:
   - Resize to 224x224 pixels
   - Normalize pixel values to [0, 1]
   - Apply data augmentation (training only)
3. **VGG16 Processing**:
   - Extract features using pre-trained VGG16
   - Fine-tuned layers for bird-specific features
4. **Classification**: Dense layers with softmax activation
5. **Post-processing**: Top-5 predictions with confidence scores

---

## 📊 Model Performance

| Model Type | Architecture | Accuracy | Input Shape | Parameters |
|------------|-------------|----------|-------------|------------|
| **Audio** | CRNN (BiLSTM + Attention) | **~90%** | (500, 33) | ~2.1M |
| **Image** | VGG16 (Transfer Learning) | **~85%** | (224, 224, 3) | ~15M |

### 📈 Training Results

![Training History](static/training_history.png)

**Key Metrics**:
- **Training Accuracy**: 95%+ (both models)
- **Validation Accuracy**: 90% (audio), 85% (image)
- **Training Time**: ~4 hours (audio), ~6 hours (image)
- **Inference Time**: <1 second per prediction

---

## 🗃 Dataset Information

### 📊 Dataset Statistics
- **Total Bird Species**: 143 species
- **Audio Samples**: 10,000+ recordings
- **Image Samples**: 8,000+ high-quality images
- **Data Sources**: 
  - Xeno-canto (audio recordings)
  - eBird (species information)
  - iNaturalist (images)
  - Custom recordings

### 🐦 Species Coverage
The model covers 143 bird species commonly found in various regions, including:
- **Songbirds**: Robins, Sparrows, Finches
- **Raptors**: Eagles, Hawks, Owls
- **Waterbirds**: Ducks, Herons, Kingfishers
- **Game Birds**: Pheasants, Quails
- **Exotic Species**: Parrots, Hornbills

> 📧 **Dataset Access**: Contact [your-email@example.com] for dataset availability.

---

## 🔧 Configuration

### Model Hyperparameters

**Audio Model (CRNN)**:
\`\`\`python
# Audio preprocessing
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Model architecture
CNN_FILTERS = [32, 64, 128]
LSTM_UNITS = 128
ATTENTION_UNITS = 64
DROPOUT_RATE = 0.3
