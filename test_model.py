import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model('crop_disease_model_v2.h5')

# Load test data
test_path = "organized_data/test"
IMG_SIZE = 224
BATCH_SIZE = 32

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path, 
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE,
    shuffle=False  # Don't shuffle for accurate evaluation
)

print("Testing model on test set...")
print("="*50)

# Evaluate
loss, accuracy = model.evaluate(test_gen, verbose=1)

print("\n" + "="*50)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Test Loss: {loss:.4f}")
print("="*50)

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# Classification report
print("\nClassification Report:")
print(classification_report(
    true_classes, 
    predicted_classes,
    target_names=list(test_gen.class_indices.keys())
))

print("\n✅ Testing complete!")
