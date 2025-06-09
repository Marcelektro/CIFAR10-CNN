# CIFAR-10 Webcam Classifier Inference

An opencv-based real-time webcam classifier using a pre-trained CIFAR-10 model.

## Requirements
Supplied by the parents `requirements.txt` file.

## Usage
Run the script:

```bash
python3 ./inference_webcam.py
```

## Use case
1. Run the script.
2. The webcam will start automatically.
   - If it doesn't, make sure the webcam is accessible by the system (e.g. `lsusb`) and not used by another application.   
        A webcam can only be used by one application at a time.
3. The model will classify the webcam feed in real-time.
4. Look at the results drawn on the webcam preview.
5. Press `q` to quit the program.
