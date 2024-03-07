import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

np.random.seed(42)
tf.random.set_seed(42)

# Define the original data directory
train_data_dir = r'C:\Users\Admin\Desktop\BharatIntern\Project 2\kagglecatsanddogs_3367a\Petimages'

# Define image dimensions
img_width, img_height = 150, 150

def load_and_preprocess_data(data_dir, num_samples=None):
    images = []
    labels = []

    for label, category in enumerate(['Cat', 'Dog']):
        category_dir = os.path.join(data_dir, category)
        for img_name in os.listdir(category_dir)[:num_samples]:
            img_path = os.path.join(category_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0  # Normalize pixel values to range [0, 1]
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load and preprocess the training data (limiting the number of samples to avoid memory error)
train_images, train_labels = load_and_preprocess_data(train_data_dir, num_samples=1000)

# Display some statistics
print("Number of training images:", len(train_images))
print("Number of cat images:", np.sum(train_labels == 0))
print("Number of dog images:", np.sum(train_labels == 1))

# Define the CNN model
model = Sequential(name="my_cnn_model")
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Since it's a binary classification problem
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(train_images, train_labels)

print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)

# Saving the model in the Keras format
model.save("my_cnn_model.keras")
print("Model saved as my_cnn_model.keras")



# Lets TEST our Model
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from tabulate import tabulate
from colorama import Fore, Style
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(data_dir):
    images = []
    labels = []

    for label, category in enumerate(['Cat', 'Dog']):
        category_dir = os.path.join(data_dir, category)
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0  # Normalize pixel values to range [0, 1]
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Define image dimensions
img_width, img_height = 150, 150

# Path to the folder containing the test images
test_dir = r"C:\Users\Admin\Desktop\BharatIntern\Project 2\kagglecatsanddogs_3367a\Petimages\Test_data"

# Load the saved model
model = load_model("my_cnn_model.keras")

# Get the list of class names (folder names)
class_names = sorted(os.listdir(test_dir))

# Initialize lists to store prediction results
predictions_table = []

# Loop through each image in the test directory
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        
        # Load and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0  # Normalize pixel values to range [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make predictions without verbose output
        predictions = model.predict(img, verbose=0)
        predicted_class = class_names[int(np.round(predictions[0]))]  # Use int(np.round(predictions[0])) to get the predicted class index
        
        # Store prediction results
        predictions_table.append([img_file, predicted_class])

# Print predictions table with color
headers = [Fore.BLUE + "Image Name", Fore.GREEN + "Predicted Class"]
print(tabulate([(Fore.BLUE + row[0], Fore.GREEN + row[1]) for row in predictions_table], headers=headers, tablefmt="fancy_grid", numalign="center"))

# Evaluate the model on the test data without verbose output
test_images, test_labels = load_and_preprocess_data(test_dir)
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"{Fore.YELLOW}Test Loss:", test_loss)
print(f"{Fore.YELLOW}Test Accuracy:", test_accuracy)

# Plotting predictions
predicted_classes = [class_names[int(np.round(predictions[0]))] for predictions in model.predict(test_images)]

# Display first 5 images with predicted output
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i])
    plt.title(f'Predicted: {predicted_classes[i]}')
    plt.axis('off')
plt.show()

# Bar plot
plt.figure(figsize=(10, 6))
plt.bar(class_names, [predicted_classes.count(cls) for cls in class_names], color='skyblue', edgecolor='black', linewidth=2)
plt.title('Predicted Classes Distribution (Bar Plot)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('bar_plot.png')
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie([predicted_classes.count(cls) for cls in class_names], labels=class_names, autopct='%1.1f%%', startangle=140)
plt.title('Predicted Classes Distribution (Pie Chart)')
plt.axis('equal')
plt.savefig('pie_chart.png')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(predicted_classes, bins=len(class_names), color='skyblue', edgecolor='black', linewidth=2)
plt.title('Predicted Classes Distribution (Histogram)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('.png')
plt.show()
