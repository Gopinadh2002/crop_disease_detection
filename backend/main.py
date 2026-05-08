from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI(title="Crop Disease Detection API")

# Enable CORS (so frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../crop_disease_model_v2.h5")
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# Image size used during training
IMG_SIZE = 224

# Disease class labels (in same order as training)
# Get these from your training data directory
CLASS_LABELS = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

print(f"Classes loaded: {len(CLASS_LABELS)}")


def preprocess_image(img_file):
    """
    Preprocess uploaded image for prediction
    """
    try:
        # Read image
        img = Image.open(io.BytesIO(img_file)).convert('RGB')
        
        # Resize to 224x224
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize (divide by 255)
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects batch of images)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        return None, str(e)


@app.get("/")
def read_root():
    """Home endpoint"""
    return {
        "message": "Crop Disease Detection API",
        "endpoint": "/api/predict",
        "usage": "POST /api/predict with image file"
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from uploaded image
    
    Input: Image file (jpg, png, etc.)
    Output: 
    {
        "disease": "Tomato_Early_blight",
        "confidence": 0.95,
        "all_predictions": {...}
    }
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Preprocess
        img_array = preprocess_image(contents)
        if img_array is None:
            return {"error": "Could not process image"}
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get disease name
        disease_name = CLASS_LABELS[predicted_class_idx]
        
        # Get all predictions with confidence
        all_predictions = {}
        for idx, label in enumerate(CLASS_LABELS):
            all_predictions[label] = float(predictions[0][idx])
        
        return {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "all_predictions": all_predictions
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict on multiple images at once
    """
    results = []
    for file in files:
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        if img_array is not None:
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            disease_name = CLASS_LABELS[predicted_class_idx]
            
            results.append({
                "filename": file.filename,
                "disease": disease_name,
                "confidence": round(confidence, 4)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
