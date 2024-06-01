from google.colab import drive
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# Load AFLW2K3D dataset
ds = tfds.load('aflw2k3d', split='train')

images = []
landmarks2D = []

# Iterate over the dataset
for ex in ds.take(500):  # Reduced to 500 examples
    images.append(tf.image.resize(ex['image'], (128, 128)))  # Resize images to (128, 128)
    landmarks2D.append(ex['landmarks_68_3d_xy_normalized'])
# Convert lists to numpy arrays
images = np.array(images)
landmarks2D = np.array(landmarks2D)

# Normalize images to range [0, 1]
images = images / 255.0

# Reshape landmarks to (num_examples, num_landmarks * 2)
landmarks2D = landmarks2D[:, :, :2].reshape(-1, 68 * 2)

# Split the dataset
train_images, test_images, train_landmarks, test_landmarks = train_test_split(
    images, landmarks2D, test_size=100, train_size=400, random_state=42  # Reduced to 400 training, 100 testing
)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(68*2)  # Output layer with shape [(68*2)*1]
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

batch_size = 16

# Train the model
model.fit(train_images, train_landmarks, epochs=10, batch_size=batch_size, validation_split=0.1)
