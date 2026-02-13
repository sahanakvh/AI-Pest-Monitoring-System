import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = 224
BATCH_SIZE = 16

TRAIN_DIR = "../dataset/train"
TEST_DIR = "../dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load Pretrained Model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=3

)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

test_data.reset()
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(test_data.classes, y_pred, target_names=list(test_data.class_indices.keys())))

print("\nConfusion Matrix:\n")
print(confusion_matrix(test_data.classes, y_pred))


model.save("pest_model.h5")

loss, accuracy = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
