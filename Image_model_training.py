import os
import numpy as np
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle  # Added for saving class indices

#Automatically skip broken images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Dataset paths
train_dir = 'E:/143_Splitted_Cleaned/train'
valid_dir = 'E:/143_Splitted_Cleaned/valid'
test_dir = 'E:/143_Splitted_Cleaned/test'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = len(os.listdir(train_dir))

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_set = test_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=EPOCHS,
    steps_per_epoch=len(train_set),
    validation_steps=len(valid_set)
)

# Save model and class indices
model.save('models/bird_classifier.h5')
with open('models/class_indices.pkl', 'wb') as f:
    pickle.dump(train_set.class_indices, f)

# Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('static/training_history.png')
plt.close()