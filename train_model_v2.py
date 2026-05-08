import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Smaller, faster model
train_path = "organized_data/train"
val_path = "organized_data/val"

IMG_SIZE = 224
BATCH_SIZE = 16

print("Loading data...")
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

print("Building model...")
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(base.input, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  # Just 5 epochs for testing
    verbose=1
)

model.save("crop_disease_model_v2.h5")
print("Done! Check accuracy below:")
print(f"Final Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"Final Val Acc: {history.history['val_accuracy'][-1]:.4f}")
