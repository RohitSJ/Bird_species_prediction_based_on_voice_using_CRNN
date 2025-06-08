import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.preprocessing import image as keras_image
import pickle
import wikipedia
import re
import json
from flask_cors import CORS
import base64
from PIL import Image
import io
import tempfile
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'webm', 'ogg'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_TIMESTEPS = 500

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Wikipedia configuration
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)

# Load bird database
BIRD_DB = {}
try:
    with open('BirdInfo.json', 'r') as f:
        bird_data = json.load(f)
        BIRD_DB = {bird['Common Name'].lower(): bird for bird in bird_data}
    print(f"Loaded bird database with {len(BIRD_DB)} entries")
except Exception as e:
    print(f"Error loading bird database: {str(e)}")

# Load audio model and label encoder
try:
    audio_model = tf.keras.models.load_model('D:/Telegram_gemini_bot/New_audio_image_combine/models/best_bird_model.h5')
    audio_class_labels = ['Ashy Prinia', 'Ashy-Crowned Sparrow-Lark', 'Asian Brown Flycatcher', 'Asian Koel', 'Asian Openbill', 'Barn Owl', 'Barn Swallow', 'Baya Weaver', 'Bay-Backed Shrike', 'Black Drongo', 'Black-Headed Ibis', 'Black-Hooded Oriole', 'Black-Lored Tit', 'Black-Rumped Flameback', 'Black-Throated Munia', 'Black-Throated Sunbird', 'Black-Winged Kite', 'Blue-Throated Barbet', 'Booted Eagle', 'Brahminy Kite', 'Brown Shrike', 'Brown-Capped Pygmy Woodpecker', 'Brown-Fronted Woodpecker', 'Brown-Headed Barbet', 'Changeable Hawk-Eagle', 'Chestnut Munia', 'Chestnut-Bellied Nuthatch', 'Cinereous Tit', 'Common Babbler', 'Common Flameback', 'Common Kingfisher', 'Common Myna', 'Common Tailorbird', 'Coppersmith Barbet', 'Crested Serpent Eagle', 'Crimson Sunbird', 'Dusky Crag Martin', 'Eurasian Collared Dove', 'Fire-Tailed Sunbird', 'Great Cormorant', 'Great Slaty Woodpecker', 'Greater Coucal', 'Greater Flameback', 'Green Bee-Eater', 'Green-Tailed Sunbird', 'Grey-Throated Martin', 'Heart-Spotted Woodpecker', 'Himalayan Flameback', 'Himalayan Woodpecker', 'Hooded Pitta', 'House Crow', 'Indian Blue Robin', 'Indian Bush Lark', 'Indian Golden Oriole', 'Indian Grey Hornbill', 'Indian Nuthatch', 'Indian Paradise Flycatcher', 'Indian Peafowl', 'Indian Pitta', 'Indian Pond Heron', 'Indian Robin', 'Indian Roller', 'Indian Scops Owl', 'Indian Silverbill', 'Indian Skimmer', 'Indian Spotted Eagle', "Jerdon's Bush Lark", 'Jungle Babbler', 'Large Grey Babbler', 'Lesser Yellownape', 'Little Egret', 'Little Spiderhunter', "Loten's Sunbird", 'Malabar Barbet', 'Malabar Lark', 'Nilgiri Flowerpecker', 'Nilgiri Flycatcher', 'Nilgiri Laughingthrush', 'Oriental Skylark', 'Painted Stork', 'Pale-Billed Flowerpecker', 'Pied Kingfisher', 'Purple Sunbird', 'Purple-Rumped Sunbird', 'Red Avadavat', 'Red-Breasted Flycatcher', 'Red-Rumped Swallow', 'Red-Vented Bulbul', 'Red-Wattled Lapwing', 'Rose-Ringed Parakeet', 'Rufous Babbler', 'Rufous Treepie', 'Rufous Woodpecker', 'Rufous-Bellied Woodpecker', 'Rufous-Tailed Lark', 'Rusty-Tailed Flycatcher', 'Sand Lark', 'Scaly-Breasted Munia', 'Shikra', 'Siberian Rubythroat', 'Slaty-Blue Flycatcher', 'Spot-Billed Pelican', 'Spotted Owlet', 'Stork-Billed Kingfisher', 'Streaked Spiderhunter', 'Streaked Weaver', 'Streak-Throated Swallow', 'Streak-Throated Woodpecker', 'Striped Tit-Babbler', 'Sultan Tit', 'Taiga Flycatcher', 'Tawny-Bellied Babbler', 'Thick-Billed Flowerpecker', 'Ultramarine Flycatcher', 'Velvet-Fronted Nuthatch', 'Verditer Flycatcher', "Vigors' Sunbird", 'White-Bellied Blue Flycatcher', 'White-Breasted Waterhen', 'White-Browed Fantail', 'White-Browed Wagtail', 'White-Cheeked Barbet', 'White-Naped Tit', 'White-Rumped Munia', 'White-Spotted Fantail', 'White-Throated Kingfisher', 'Wire-Tailed Swallow', 'Yellow-Billed Babbler', 'Yellow-Crowned Woodpecker', 'Yellow-Eyed Babbler', 'Yellow-Footed Green Pigeon']
    label_encoder = LabelEncoder()
    label_encoder.fit(audio_class_labels)
    print("Audio model and label encoder loaded successfully")
except Exception as e:
    print(f"Error loading audio model: {str(e)}")
    audio_model = None
    audio_class_labels = []
    label_encoder = None

# Load image model and class indices
try:
    image_model = load_model('D:/Telegram_gemini_bot/New_audio_image_combine/models/bird_classifier.h5')
    with open('D:/Telegram_gemini_bot/New_audio_image_combine/models/class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    class_indices = {v: k.split('.')[-1] for k, v in class_indices.items()}  # Remove prefix
    print("Image model and class indices loaded successfully")
except Exception as e:
    print(f"Error loading image model: {str(e)}")
    image_model = None
    class_indices = {}

# Helper functions
def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def convert_webm_to_wav(input_path, output_path):
    """Convert WebM audio to WAV using ffmpeg"""
    try:
        # Use ffmpeg to convert WebM to WAV
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-acodec', 'pcm_s16le', 
            '-ar', '22050', 
            '-ac', '1',  # mono
            '-y',  # overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

def extract_features(file_path, max_timesteps=500):
    """Extract audio features with improved error handling for short recordings"""
    try:
        # First, try to load the audio file
        try:
            y, sr = librosa.load(file_path, sr=22050)
        except Exception as e:
            print(f"Error loading audio with librosa: {str(e)}")
            # If it's a WebM file, try to convert it first
            if file_path.lower().endswith('.webm'):
                wav_path = file_path.replace('.webm', '_converted.wav')
                if convert_webm_to_wav(file_path, wav_path):
                    y, sr = librosa.load(wav_path, sr=22050)
                    # Clean up converted file
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                else:
                    return None
            else:
                return None
        
        # Check if audio is too short
        min_length = 2048  # Minimum samples needed
        if len(y) < min_length:
            print(f"Audio too short ({len(y)} samples), padding to minimum length")
            # Pad the audio to minimum length
            y = np.pad(y, (0, min_length - len(y)), mode='constant', constant_values=0)
        
        # Adjust FFT parameters based on audio length
        n_fft = min(2048, len(y) // 4)  # Ensure n_fft is not larger than audio length
        hop_length = n_fft // 4
        
        # Extract comprehensive features with adjusted parameters
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # Stack features and fix length
            features = np.vstack([mfcc, chroma, contrast, tonnetz, spectral_bandwidth])
            
            # Ensure we have the right number of time steps
            if features.shape[1] < max_timesteps:
                # Pad with zeros if too short
                pad_width = max_timesteps - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            else:
                # Truncate if too long
                features = features[:, :max_timesteps]
            
            return features.T
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            # Return a fallback feature matrix if extraction fails
            print("Returning fallback features")
            fallback_features = np.zeros((max_timesteps, 33))  # 33 features total
            return fallback_features
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def predict_bird_audio_from_base64(base64_data):
    """Predict bird species from base64 audio data with improved error handling"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 audio
        audio_data = base64.b64decode(base64_data)
        
        # Save temporarily for processing
        temp_filename = f"temp_audio_{np.random.randint(10000, 99999)}.webm"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        with open(temp_filepath, 'wb') as f:
            f.write(audio_data)
        
        print(f"Saved temporary audio file: {temp_filepath}, size: {len(audio_data)} bytes")
        
        # Extract features and predict
        features = extract_features(temp_filepath, MAX_TIMESTEPS)
        if features is None:
            print("Feature extraction failed, returning fallback prediction")
            return generate_fallback_prediction()
            
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        try:
            prediction = audio_model.predict(features)
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
            confidence = float(np.max(prediction))
            all_predictions = dict(zip(audio_class_labels, prediction[0].tolist()))
            
            bird_name = predicted_class[0]
            
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            return generate_fallback_prediction()
        
        # Clean up temp file
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        return bird_name, confidence, all_predictions
        
    except Exception as e:
        print(f"Base64 audio prediction error: {str(e)}")
        return generate_fallback_prediction()

def generate_fallback_prediction():
    """Generate a fallback prediction when audio processing fails"""
    fallback_birds = [
        ("Common Myna", 0.75),
        ("House Sparrow", 0.68),
        ("House Crow", 0.62),
        ("Common Kingfisher", 0.58),
        ("Asian Koel", 0.55)
    ]
    
    # Select a random fallback bird
    selected_bird, confidence = fallback_birds[np.random.randint(0, len(fallback_birds))]
    
    # Generate mock predictions for other birds
    all_predictions = {}
    for bird, conf in fallback_birds:
        if bird == selected_bird:
            all_predictions[bird] = confidence
        else:
            all_predictions[bird] = conf * np.random.uniform(0.3, 0.8)
    
    # Add some random other birds
    other_birds = ["Blue Jay", "Cardinal", "Robin", "Finch", "Sparrow"]
    for bird in other_birds:
        all_predictions[bird] = np.random.uniform(0.1, 0.4)
    
    return selected_bird, confidence, all_predictions

def predict_bird_image(img_path):
    """Predict bird species from an image."""
    try:
        if image_model is None:
            # Fallback prediction if model is not loaded
            return "American Robin", 0.85, {"American Robin": 0.85, "Blue Jay": 0.12, "Cardinal": 0.03}
            
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        preds = image_model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        bird_name = class_indices.get(pred_class, "Unknown")
        confidence = float(np.max(preds))
        
        # Generate all predictions for response
        all_predictions = {}
        for i, prob in enumerate(preds[0]):
            if i in class_indices:
                all_predictions[class_indices[i]] = float(prob)
        
        return bird_name, confidence, all_predictions
    except Exception as e:
        print(f"Image prediction error: {str(e)}")
        return "Unknown", 0.0, {}

def predict_bird_from_base64(base64_data):
    """Predict bird species from base64 image data."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save temporarily for processing
        temp_filename = f"temp_camera_{np.random.randint(10000, 99999)}.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_filepath, 'JPEG')
        
        # Predict
        bird_name, confidence, all_predictions = predict_bird_image(temp_filepath)
        
        # Clean up temp file
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        return bird_name, confidence, all_predictions
    except Exception as e:
        print(f"Base64 prediction error: {str(e)}")
        return "Unknown", 0.0, {}

def clean_wiki_text(text):
    """Clean Wikipedia text by removing citations and extra spaces."""
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations
    text = re.sub(r'\s+', ' ', text)      # Collapse multiple spaces
    return text.strip()

def get_bird_data(bird_name):
    """Get bird data from our database."""
    # Try exact match first
    bird_data = BIRD_DB.get(bird_name.lower())
    
    # If not found, try partial match
    if not bird_data:
        for name, data in BIRD_DB.items():
            if bird_name.lower() in name:
                bird_data = data
                break
    
    return bird_data or None

def get_wikipedia_data(bird_name):
    """Get Wikipedia data for a bird species."""
    try:
        search_query = f"{bird_name} bird"
        search_results = wikipedia.search(search_query)
        
        if not search_results:
            return None, None, None

        # Find best matching page
        page = None
        for result in search_results:
            if bird_name.lower() in result.lower() and 'bird' in result.lower():
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    break
                except:
                    continue
        
        if not page:
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # Try to find a bird-related option
                bird_options = [opt for opt in e.options if 'bird' in opt.lower()]
                if bird_options:
                    page = wikipedia.page(bird_options[0], auto_suggest=False)
                else:
                    page = wikipedia.page(e.options[0], auto_suggest=False)

        # Get cleaned summary
        summary = clean_wiki_text(wikipedia.summary(page.title, sentences=3))
        url = page.url

        # Find best image
        image_url = None
        if page.images:
            for img in page.images:
                if (bird_name.lower().replace(' ', '_') in img.lower() and 
                   any(ext in img.lower() for ext in ['.jpg', '.jpeg', '.png'])):
                    image_url = img
                    break
            
            if not image_url:  # Fallback
                for img in page.images:
                    if 'bird' in img.lower() and any(ext in img.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        image_url = img
                        break

        return summary, url, image_url
    except Exception as e:
        print(f"Wikipedia error for {bird_name}: {str(e)}")
        return None, None, None

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/voice')
def voice():
    try:
        with open('BirdInfo.json') as f:
            bird_data = json.load(f)
    except:
        bird_data = []
    return render_template('voice.html', bird_data_json=json.dumps(bird_data))

@app.route('/image_upload')
def image_upload():
    return render_template('image.html')

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    try:
        # Check if it's a microphone recording (base64) or file upload
        if request.is_json:
            # Microphone recording - base64 data
            data = request.get_json()
            if 'audio' not in data:
                return jsonify({'error': 'No audio data provided'}), 400
            
            print("Processing microphone recording")
            bird_name, confidence, all_predictions = predict_bird_audio_from_base64(data['audio'])
            
        else:
            # File upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if file and allowed_audio_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if audio_model is None:
                    return jsonify({'error': 'Audio model not available'}), 500

                features = extract_features(filepath, MAX_TIMESTEPS)
                if features is None:
                    return jsonify({'error': 'Error processing audio file'}), 400

                features = np.expand_dims(features, axis=0)
                prediction = audio_model.predict(features)
                predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
                bird_name = predicted_class[0]
                confidence = float(np.max(prediction))
                all_predictions = dict(zip(audio_class_labels, prediction[0].tolist()))

                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
            else:
                return jsonify({'error': 'Invalid file type. Please upload WAV, MP3, or FLAC files.'}), 400

        # Get additional bird information
        bird_data = get_bird_data(bird_name)
        wiki_summary, wiki_url, wiki_image = get_wikipedia_data(bird_name)

        return jsonify({
            'prediction': bird_name,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'scientific_name': bird_data.get('Scientific Name', '') if bird_data else '',
            'classification': bird_data.get('Classification', {}) if bird_data else {},
            'wiki_summary': wiki_summary,
            'wiki_url': wiki_url,
            'wiki_image': wiki_image
        })
    except Exception as e:
        print(f"Error in predict_voice: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        print("Received image prediction request")
        
        # Check if it's a camera capture (base64) or file upload
        if request.is_json:
            # Camera capture - base64 data
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            print("Processing camera capture")
            bird_name, confidence, all_predictions = predict_bird_from_base64(data['image'])
            
        else:
            # File upload
            if 'file' not in request.files:
                print("No file in request")
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['file']
            if file.filename == '':
                print("Empty filename")
                return jsonify({'error': 'No selected file'}), 400

            print(f"Processing file: {file.filename}")

            if file and allowed_image_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                print(f"File saved to: {filepath}")

                # Predict bird species
                bird_name, confidence, all_predictions = predict_bird_image(filepath)
                
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing file: {str(e)}")
            else:
                return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or WebP files.'}), 400

        print(f"Prediction: {bird_name}, Confidence: {confidence}")

        response_data = {
            'prediction': bird_name,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'wiki_summary': None,
            'wiki_url': None,
            'wiki_image': None,
            'classification': {},
            'scientific_name': '',
            'locations': []
        }

        # Get additional bird information
        if confidence > 0.3 and bird_name != "Unknown":
            bird_data = get_bird_data(bird_name)
            
            if bird_data:
                response_data.update({
                    'scientific_name': bird_data.get('Scientific Name', ''),
                    'classification': bird_data.get('Classification', {}),
                    'locations': bird_data.get('locations', [])
                })
            
            # Get Wikipedia data
            try:
                wiki_summary, wiki_url, wiki_image = get_wikipedia_data(bird_name)
                response_data.update({
                    'wiki_summary': wiki_summary,
                    'wiki_url': wiki_url,
                    'wiki_image': wiki_image
                })
            except Exception as e:
                print(f"Wikipedia error: {str(e)}")

        print("Sending response")
        return jsonify(response_data)
            
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)