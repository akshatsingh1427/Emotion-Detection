import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

print("Loading data...")
data = pd.read_csv('data/fer2013.csv')

pixels = data['pixels'].tolist()
emotions = data['emotion'].values

images = []
for pixel_seq in pixels:
    img = np.array(pixel_seq.split(), dtype='float32').reshape(48, 48)
    images.append(img)

images = np.array(images) / 255.0
images = np.expand_dims(images, -1)  # Add channel dimension

labels = keras.utils.to_categorical(emotions, 7)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("Training model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=64)

# Save model
model.save('emotion_model.h5')
print("Model saved as 'emotion_model.h5'")

# Plot results
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()
