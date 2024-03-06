import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

np.random.seed(42)
tf.random.set_seed(42)

# Define the original data directory
train_data_dir = r'C:\Users\Admin\Desktop\BharatIntern\Project 2\kagglecatsanddogs_3367a\Petimages'

# Define image dimensions
img_width, img_height = 150, 150

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


# Load and preprocess the training data
train_images, train_labels = load_and_preprocess_data(train_data_dir)

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


#Now we can test our model 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the model we saved
model = load_model("my_cnn_model.keras")

# Path to the folder containing the test images
test_dir = r"C:\Users\Admin\Desktop\BharatIntern\Project 2\kagglecatsanddogs_3367a\Petimages\Test_data"

# Get the list of class names (folder names)
class_names = sorted(os.listdir(test_dir))
# Load and preprocess the test data
test_images, test_labels = load_and_preprocess_data(test_dir)

# Loop through each image in the test directory
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(img_width, img_height))  # Adjust target_size to match model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_names[int(np.round(predictions[0]))]  # Use int(np.round(predictions[0])) to get the predicted class index
        
        print(f"Predicted class: {predicted_class}")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
