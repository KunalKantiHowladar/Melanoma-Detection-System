import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Skin Cancer Detection')
    parser.add_argument('--image', required=True, help='Path to the image')
    args = parser.parse_args()
    model = load_model('skin_cancer_model.h5')
    # Use sorted folder names for consistent class ordering
    classes = sorted([d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))])
    img = preprocess_image(args.image)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = float(pred[0][class_idx])
    print(f"Predicted class: {classes[class_idx]} (Confidence: {confidence:.2f})")
