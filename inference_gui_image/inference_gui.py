import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# disable GPU so we can train/play with the model in the notebook with GPU, and this lightweight app can run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(img_path):
    img = Image.open(img_path).resize((32, 32)).convert('RGB') # rgb format 32x32
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path).resize((128, 128))
    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    processed = preprocess_image(file_path)
    prediction = model.predict(processed)
    class_idx = np.argmax(prediction)
    result_label.config(text=f"Predicted: {classes[class_idx]}")


if __name__ == "__main__":

    model = load_model('../models/cifar10_cnn_model_advanced.keras')

    root = tk.Tk()
    root.title("CIFAR-10 Image Classifier")

    frame = tk.Frame(root)
    frame.pack()

    image_label = tk.Label(frame)
    image_label.pack()

    btn = tk.Button(frame, text="Choose Image", command=classify_image)
    btn.pack()

    result_label = tk.Label(frame, text="Prediction will appear here", font=('Arial', 14))
    result_label.pack()

    root.mainloop()
