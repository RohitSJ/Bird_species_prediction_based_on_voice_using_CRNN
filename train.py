import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_DIR = 'D:/Telegram_gemini_bot/New_project/Cleaned_Dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VALID_DIR = os.path.join(DATASET_DIR, 'val')

MAX_TIMESTEPS = 500
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 3e-4
NUM_FEATURES = 64  # Optimized feature size

# Enhanced feature extraction
def extract_features(file_path, max_timesteps=500):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract comprehensive features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=512)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)
        
        # Stack features and fix length
        features = np.vstack([mfcc, chroma, contrast, tonnetz, spectral_bandwidth])
        features = librosa.util.fix_length(features, size=max_timesteps, axis=1)
        
        return features.T
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_dataset(dataset_path, max_timesteps, label_encoder):
    features = []
    labels = []
    
    for bird_class in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, bird_class)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing {bird_class}...")
        for file in tqdm(os.listdir(class_path)):
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_path, file)
                feature = extract_features(file_path, max_timesteps)
                if feature is not None:
                    features.append(feature)
                    labels.append(bird_class)
    
    features = np.array(features)
    labels = np.array(labels)
    
    if not hasattr(label_encoder, 'classes_'):
        label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    
    return features, encoded_labels, label_encoder.classes_

# Enhanced CRNN Model with Attention
def create_crnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Conv Blocks with residual connections
    x = layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Bidirectional LSTM with Attention
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    attention = layers.Attention()([x, x])
    x = layers.Concatenate()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# CNN Model for Ensemble
def create_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# Ensemble Model
class EnsembleModel(tf.keras.Model):
    def __init__(self, models, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.avg = layers.Average()
        self.final_dense = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        outputs = [model(inputs) for model in self.models]
        averaged = self.avg(outputs)
        return self.final_dense(averaged)

def train_and_evaluate(model, train_ds, val_ds, test_ds, model_name):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(f'best_{model_name}.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    print(f"/nTraining {model_name}...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"/nEvaluating {model_name} on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

def main():
    label_encoder = LabelEncoder()
    
    print("Loading datasets...")
    X_train, y_train, class_labels = load_dataset(TRAIN_DIR, MAX_TIMESTEPS, label_encoder)
    X_val, y_val, _ = load_dataset(VALID_DIR, MAX_TIMESTEPS, label_encoder)
    X_test, y_test, _ = load_dataset(TEST_DIR, MAX_TIMESTEPS, label_encoder)
    
    # Print dataset information
    print(f"/nDataset shapes:")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Validation: {X_val.shape}, {y_val.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    print(f"Number of classes: {len(class_labels)}")
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(class_labels)
    
    # Train individual models
    crnn = create_crnn_model(input_shape, num_classes)
    cnn = create_cnn_model(input_shape, num_classes)
    
    print("/nTraining CRNN model...")
    crnn_acc = train_and_evaluate(crnn, train_ds, val_ds, test_ds, 'crnn')
    
    print("/nTraining CNN model...")
    cnn_acc = train_and_evaluate(cnn, train_ds, val_ds, test_ds, 'cnn')
    
    # Create and evaluate ensemble
    ensemble = EnsembleModel([crnn, cnn], num_classes)
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("/nEvaluating Ensemble model...")
    ensemble_loss, ensemble_acc = ensemble.evaluate(test_ds)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Save best model
    if ensemble_acc > max(crnn_acc, cnn_acc):
        ensemble.save('best_bird_model.h5')
        print("/nSaved Ensemble model as best model")
    elif crnn_acc > cnn_acc:
        crnn.save('best_bird_model.h5')
        print("/nSaved CRNN model as best model")
    else:
        cnn.save('best_bird_model.h5')
        print("/nSaved CNN model as best model")

if __name__ == "__main__":
    main()