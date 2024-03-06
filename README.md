# <span style="color:blue">Image Classifier 🖼</span>

This project implements a convolutional neural network (CNN) to classify images of cats and dogs. The model is trained on the Kaggle Cats vs Dogs dataset.

## <span style="color:green">Overview</span>

The goal of this project is to build a machine learning model that can accurately classify whether an input image contains a cat or a dog. The model architecture is a CNN built using TensorFlow and Keras.
## <span style="color:green">Sample Images</span>

<img src="https://lh6.ggpht.com/EoH-UNwa_XRNuk0jB7UQDVQdTSigPu4B5zrb3I5iXk289KYfGZOzJgfwP8Y9DEA1O_k=h900" alt="Cat Image" width="200" height="150">
<img src="https://th.bing.com/th/id/R.03f137aa3456f51c9f62d9e876e7a9ff?rik=tMY5NiYPQWnnkg&riu=http%3a%2f%2fgetwallpapers.com%2fwallpaper%2ffull%2fd%2f1%2fe%2f1129346-cute-cats-wallpaper-1080x1920-full-hd.jpg&ehk=%2fJcdRdco6dASg87GmgQEx5VyEz7bYP%2fM%2bGHwd2mq20g%3d&risl=&pid=ImgRaw&r=0" alt="Dog Image" width="200" height="150">

## <span style="color:green">Dataset</span>

The dataset used for training and testing the model is the Kaggle Cats vs Dogs dataset, which consists of thousands of images of cats and dogs. The dataset is divided into training and test sets, with a balanced number of images for each class.

## <span style="color:green">Model Training</span>

The CNN model is trained using the training set of images. The training process involves feeding the images through the model, computing the loss function, and updating the model's weights using backpropagation. Training progress is monitored using metrics such as loss and accuracy.

## <span style="color:green">Model Evaluation</span>

After training, the model is evaluated using the test set of images to assess its performance on unseen data. Evaluation metrics such as accuracy are used to measure the model's performance.

## <span style="color:green">How to Use</span>

1. Clone the repository:

   ```bash
   git clone https://github.com/SyedFahad7/Image-Classifier
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the Kaggle Cats vs Dogs dataset and organize it in the appropriate directory structure.
4. link to the Kaggle Dataset : https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

5. Train the model:

   ```bash
   python train.py
   ```

6. Test the model:

   ```bash
   python test.py
   ```

7. Deploy the model (optional):

   You can deploy the trained model for inference in your own applications.

## <span style="color:green">Files</span>

- `train.py`: Script for training the CNN model.
- `test.py`: Script for testing the trained model on new images.
- `my_cnn_model.Keras`: Trained CNN model saved in HDF5 format.
- `README.md`: This file providing an overview of the project.

## <span style="color:green">🚀 About Me</span>
I'm a full stack Web & App Developer and an undergrad Data Science Student 👨‍💻🙌

## <span style="color:green">Authors</span>

- [@Fahad](https://github.com/SyedFahad7)

```
