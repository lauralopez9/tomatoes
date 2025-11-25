# === IMPORTACIONES NECESARIAS ===
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Importamos variables esenciales desde entrenamiento.py
from entrenamiento import IMG_SIZE, num_classes, train_generator, val_generator

# Importamos utilidades desde datos.py
from datos import plot_training, evaluate_model, get_random_val_image, predict_single_image


# === MODELO RESNET50 ===
base_resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_resnet.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_resnet(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

resnet_model = tf.keras.Model(inputs, outputs, name="ResNet50_tomates")

resnet_model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

resnet_model.summary()

# Entrenar
EPOCHS_RESNET = 10

history_resnet = resnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_RESNET,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Guardar modelo
resnet_model.save("modelo_tomates_resnet50.h5")
print("Modelo ResNet50 guardado como modelo_tomates_resnet50.h5")

# Gráficas de entrenamiento
plot_training(history_resnet, title="ResNet50")

# Evaluación y métricas
evaluate_model(resnet_model, name="ResNet50")

# Predicción de ejemplo
random_img = get_random_val_image()
predict_single_image(resnet_model, random_img, title_model="ResNet50")
