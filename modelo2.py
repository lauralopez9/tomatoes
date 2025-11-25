import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from entrenamiento import IMG_SIZE, num_classes, train_generator, val_generator
from datos import plot_training, evaluate_model, get_random_val_image, predict_single_image

# === EfficientNetB0 ===
base_efficient = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_efficient.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_efficient(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

efficient_model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_tomates")

efficient_model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

efficient_model.summary()

EPOCHS_EFFICIENT = 10

history_efficient = efficient_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_EFFICIENT,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Guardar modelo
efficient_model.save("modelo_tomates_efficientnetb0.h5")
print("Modelo EfficientNetB0 guardado como modelo_tomates_efficientnetb0.h5")

# Gráficas
plot_training(history_efficient, title="EfficientNetB0")

# Evaluación
evaluate_model(efficient_model, name="EfficientNetB0")

# Predicción ejemplo
random_img = get_random_val_image()
predict_single_image(efficient_model, random_img, title_model="EfficientNetB0")
