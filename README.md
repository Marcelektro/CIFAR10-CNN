# CIFAR-10 Image Classifier

A project for training and inference of a CIFAR-10 convolutional neural network (CNN) using Keras.

## Requirements
Supplied by the `requirements.txt` file.

## Usage

### Training

View `notebooks/train_cifar10_model.ipynb` for a Jupyter notebook that trains a CIFAR-10 model and saves it to a .keras file.  
Currently, ends with a model accuracy of 0.90 on the test set.

### Inference

View `inference_*` directories for different inference methods:
- `inference_webcam`: A real-time webcam classifier using OpenCV.
- `inference_gui_image`: A simple GUI for classifying images using a pre-trained CIFAR-10 model.


The docs for each inference method are in their respective `README.md` files.

