# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split

# importing sample data from a dataset
crocodilian_dir = 'crocodilian_images'

# names of outcome classes
class_names = ['crocodile', 'alligator', 'caiman', 'gharial']

# Define image data generator with data splitting
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split dataset into training and validation sets
)

# Load and preprocess crocodilian images
generator = datagen.flow_from_directory(
    crocodilian_dir,
    target_size=(256, 256),   
    batch_size=20,
    class_mode='categorical',  # Use categorical labels
    shuffle=True             # Shuffle images for training
)

# Extract images and labels from generator
train_images, train_labels = [], []
for _ in range(len(generator)):
    images, labels = next(generator)
    train_images.extend(images)
    train_labels.extend(labels)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Split data into training and validation sets
train_images, validation_images, train_labels, validation_labels = train_test_split(
    train_images,
    train_labels,
    test_size=0.2,  # Adjust test size as needed
    stratify=train_labels,  # Preserve class distribution
    random_state=42  # Set random state for reproducibility
)

# Further split validation set into validation and testing sets
validation_images, test_images, validation_labels, test_labels = train_test_split(
    validation_images,
    validation_labels,
    test_size=0.5,  # 50% of validation set for testing
    stratify=validation_labels,
    random_state=42
)

# Machine learning model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),  # Input layer with shape (256, 256, 3)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4)  # Adjust the number of units to match the number of classes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# creates the probability model
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# some sample predictions
predictions = probability_model.predict(test_images)

# plotting functions to display results
def plot_image(predictions_array, true_label, img, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
                                          color=color)

def plot_value_array(predictions_array, true_label, class_names):
    plt.grid(False)
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Prompt for file path
while True:
    file_path = input("Enter the file path or filename of the image (or type 'exit' to quit): ")

    if file_path.lower() == 'exit':
        break  # Exit the loop if the user types 'exit'

    if os.path.exists(file_path):
        # Load and preprocess the image
        img = cv2.imread(file_path)  # Load as grayscale
        img = cv2.resize(img, (256, 256))  # Resize to match model input shape
        img = img / 255.0  # Normalize pixel values

        # Reshape image to match model input shape
        img = np.expand_dims(img, axis=0)

        # Classify the image
        predictions = probability_model.predict(img)

        if isinstance(predictions[0], np.ndarray) and len(predictions[0]) == 10:
            predictions_array = predictions[0]

        # Ensure img is a single image array
        if isinstance(img, np.ndarray) and len(img) > 0:
            img = img[0]

        # Extract true class label from the file name
        file_name = os.path.basename(file_path)
        class_name = file_name.split('_')[0]  # Extract class name from filename
        true_label = class_names.index(class_name)  # Get the index of the class name in class_names

        # Display results with true class name
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(predictions[0], true_label, img, class_names) 
        plt.subplot(1, 2, 2)
        plot_value_array(predictions[0], true_label, class_names)  
        plt.show()
    else:
        print("Invalid file path!!!")