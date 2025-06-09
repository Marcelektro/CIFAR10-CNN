import os
import cv2
import numpy as np
from keras.models import load_model

# disable GPU so we can train/play with the model in the notebook with GPU, and this lightweight app can run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_frame(frame):
    img = cv2.resize(frame, (32, 32))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def process_frame(frame):
    return model.predict(preprocess_frame(frame))[0]


if __name__ == "__main__":

    model = load_model('../models/cifar10_cnn_model_advanced.keras')

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Failed to open webcam.")
        exit()

    print(f"Webcam properties: Width: {frame_width}, Height: {frame_height}")

    while True:
        cap_ret, cap_frame = cap.read()
        if not cap_ret:
            break

        preds = process_frame(cap_frame)

        top_class_idx = np.argmax(preds)
        label = f"{classes[top_class_idx]} ({preds[top_class_idx]*100:.2f}%)"

        cv2.putText(cap_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2, cv2.LINE_AA)

        sorted_indices = np.argsort(preds)[::-1]
        num_classes = len(classes)
        for rank, i in enumerate(sorted_indices):
            prob = preds[i]
            green = int(255 * prob)
            red = 255 - green
            color = (100, green, red)
            text = f"{classes[i]}: {prob*100:.1f}%"
            cv2.putText(cap_frame, text, (10, 60 + rank * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('CIFAR-10 Real-time Classification', cap_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
