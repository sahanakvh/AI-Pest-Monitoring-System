import tensorflow as tf
import numpy as np
import cv2
import sys

model = tf.keras.models.load_model("pest_model.h5")

class_names = ['Early_blight', 'Late_blight', 'Septoria_leaf_spot', 'healthy']

IMG_SIZE = 224

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\nPrediction: {class_names[class_index]}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    image_path = sys.argv[1]
    predict_image(image_path)
