import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from entrenamiento import IMG_SIZE, num_classes, train_generator, val_generator
from datos import plot_training, evaluate_model, get_random_val_image, predict_single_image

# === DenseNet121 ===
base_densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_densenet.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_densenet(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

densenet_model = tf.keras.Model(inputs, outputs, name="DenseNet121_tomates")

densenet_model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

densenet_model.summary()

EPOCHS_DENSENET = 10

history_densenet = densenet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_DENSENET,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

densenet_model.save("modelo_tomates_densenet121.h5")
print("Modelo DenseNet121 guardado como modelo_tomates_densenet121.h5")

# Gráficas
plot_training(history_densenet, title="DenseNet121")

# Evaluación
evaluate_model(densenet_model, name="DenseNet121")

# Predicción ejemplo
random_img = get_random_val_image()
predict_single_image(densenet_model, random_img, title_model="DenseNet121")
