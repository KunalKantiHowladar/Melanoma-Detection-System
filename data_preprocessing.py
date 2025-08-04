import os
import cv2
import numpy as np

def load_images_from_folder(folder, img_size=(224, 224)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        # Recursively walk through all subfolders
        for root, _, files in os.walk(label_folder):
            for filename in files:
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    images, labels = load_images_from_folder('data')
    print(f"Loaded {len(images)} images.")
