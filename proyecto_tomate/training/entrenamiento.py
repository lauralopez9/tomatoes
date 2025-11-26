import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix
import itertools

print("Versión de TensorFlow:", tf.__version__)

# Tamaño de imagen y batch
IMG_SIZE = (180, 180)
BATCH_SIZE = 32

TRAIN_DIR = r"C:/Users/ll529/OneDrive/Documentos/modelado_datos/dataset/dataset/train"
VAL_DIR   = r"C:/Users/ll529/OneDrive/Documentos/modelado_datos/dataset/dataset/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
print("Número de clases:", num_classes)
print("Clases detectadas:", class_indices)

idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]
print("Orden de clases:", class_names)