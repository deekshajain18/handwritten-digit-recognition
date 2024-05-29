# Handwritten Digit Recognition

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The project is divided into two parts: training the model and using the trained model to recognize digits from a live video feed.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- tensorflow
- keras
- numpy
- opencv-python
- scikit-image

You can install the required libraries using the following command:
 pip install tensorflow keras numpy opencv-python scikit-image

## Dataset
The dataset used for training the model is the MNIST dataset, which is available in the "tensorflow.keras.datasets module".

## Training the Model
The training script train.py builds and trains a CNN on the MNIST dataset. The model architecture is as follows:

Conv2D layer with 32 filters, kernel size (3, 3), ReLU activation
Conv2D layer with 64 filters, kernel size (3, 3), ReLU activation
MaxPooling2D layer with pool size (2, 2)
Dropout layer with rate 0.25
Flatten layer
Dense layer with 128 units, ReLU activation
Dropout layer with rate 0.5
Dense layer with 10 units, softmax activation
Running the Training Script
To train the model, run the following command:

sh
Copy code
python train.py
The trained model is saved as mymodel2.h5.

## Recognizing Digits from Live Video
The recognition script digitrecognization.py uses OpenCV to capture video from the webcam and the trained model to predict the digits.

## Running the Recognition Script
To start recognizing digits from the live video feed, run the following command:
Copy code
python digitrecognization.py
Project Structure
train.py: Script to train the CNN model on the MNIST dataset.
digitrecognization.py: Script to recognize handwritten digits from a live video feed using the trained model.
mymodel2.h5: The trained CNN model (generated after running train.py).

## How It Works
The train.py script loads the MNIST dataset, normalizes the images, and converts the labels to categorical format.
It then defines the CNN model architecture, compiles the model, and trains it on the training data.
After training, the model is saved to mymodel2.h5.
The digitrecognization.py script loads the trained model and initializes the webcam.
It captures each frame, processes the image to make it suitable for the model, and predicts the digit.
The predicted digit is displayed on the original frame, and the live video feed is shown with the predicted digit overlaid.
