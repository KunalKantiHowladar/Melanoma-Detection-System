import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import build_model
from data_preprocessing import load_images_from_folder

if __name__ == "__main__":
    images, labels = load_images_from_folder('data')
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    images = images / 255.0
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(le.classes_))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save('skin_cancer_model.h5')
