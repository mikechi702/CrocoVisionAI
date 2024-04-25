# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# importing sample data from a dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# separates data into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# names of outcome classes
class_names = ['T-Shirt', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-Boot']

# Resize images to 56x56
train_images_resized = np.array([cv2.resize(img, (56, 56)) for img in train_images])
test_images_resized = np.array([cv2.resize(img, (56, 56)) for img in test_images])

# Normalize pixel values
train_images_resized = train_images_resized / 255.0
test_images_resized = test_images_resized / 255.0

# Machine learning model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(56, 56)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# beginning of training and testing the model using Epochs
model.fit(train_images_resized, train_labels, epochs=10, validation_split=0.2)
test_loss, test_acc = model.evaluate(test_images_resized,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# creates the probability model
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# some sample predictions
predictions = probability_model.predict(test_images_resized)

# plotting functions to display results
def plot_image(predictions_array, true_label, img):
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

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
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
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img = cv2.resize(img, (56, 56))  # Resize to match model input shape
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
        plot_image(predictions[0], true_label, img)  # Pass true_label instead of np.argmax(predictions[0])
        plt.subplot(1, 2, 2)
        plot_value_array(predictions[0], true_label)  # Pass true_label instead of np.argmax(predictions[0])
        plt.show()
    else:
        print("Invalid file path!!!")

# PROJECT DUE APRIL 24TH!!!
# Do not forget to keep separate user data and training data.