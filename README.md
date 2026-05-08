# 🌾 Crop Disease Detection

A deep learning project to detect crop diseases from images using a trained ResNet model, FastAPI backend, and React frontend.

## 📋 Project Overview

This application helps farmers and agricultural experts identify crop diseases by uploading plant/leaf images. The system uses a deep learning model trained on 20,000+ images of 15 different crop diseases.

**Accuracy:** 92% on test data

## 🎯 Features

- ✅ Disease detection for 15 crop disease classes
- ✅ Real-time predictions with confidence scores
- ✅ Top 5 predictions displayed
- ✅ Clean, user-friendly web interface
- ✅ RESTful API for integration

## 📊 Supported Crops & Diseases

**Pepper:**
- Bacterial Spot
- Healthy

**Potato:**
- Early Blight
- Late Blight
- Healthy

**Tomato:**
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

## 🏗️ Project Structure

crop-disease-detection/
├── backend/                    # FastAPI backend
│   ├── main.py                 # Main API file
│   ├── requirements.txt         # Python dependencies
│   ├── Procfile                 # Deployment config
│   └── runtime.txt              # Python version
├── frontend/                   # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── train_model_v2.py          # Model training script
├── test_model.py              # Model testing script
├── crop_disease_model_v2.h5   # Trained model
└── README.md

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 14+
- npm

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/Gopinadh2002/crop_disease_detection.git
cd crop_disease_detection
```

#### 2. Download Model Weights

The trained model is required to run the project:

**Option A: Google Drive** (Recommended)
- [Download crop_disease_model_v2.h5](https://drive.google.com/file/d/YOUR_DRIVE_ID/view?usp=sharing)
- Place it in the root folder

**Option B: Hugging Face Hub**
- Coming soon

**Option C: Build from scratch**
- See "Training" section below

#### 3. Setup Backend

```bash
cd backend
pip install -r requirements.txt
cd ..
```

#### 4. Setup Frontend

```bash
cd frontend
npm install
cd ..
```

## 💻 Running Locally

### Terminal 1: Start Backend

```bash
cd backend
python main.py
```

Backend will run on: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

### Terminal 2: Start Frontend

```bash
cd frontend
npm start
```

Frontend will run on: `http://localhost:3000`

### Access Application

Open browser: `http://localhost:3000`

## 📡 API Endpoints

### Single Prediction

POST /api/predict
Content-Type: multipart/form-data
Parameters:

file: Image file (jpg, png, etc.)

Response:
{
"disease": "Tomato_Early_blight",
"confidence": 0.92,
"all_predictions": {
"Tomato_Early_blight": 0.92,
"Tomato_Late_blight": 0.05,
...
}
}

### Batch Predictions

POST /api/predict-batch
Content-Type: multipart/form-data
Parameters:

files: Multiple image files

Response:
{
"results": [
{
"filename": "image1.jpg",
"disease": "Tomato_Early_blight",
"confidence": 0.92
},
...
]
}

## 🧠 Model Details

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Training Data:** PlantVillage Dataset (20,639 images)
- **Classes:** 15 disease/healthy classes
- **Input Size:** 224x224 pixels
- **Framework:** TensorFlow 2.15
- **Accuracy:** 92% (validation), 92.63% (test)

## 📚 Training

To train the model from scratch:

### 1. Prepare Data

```bash
python split_dataset.py
```

### 2. Train Model

```bash
python train_model_v2.py
```

Training takes ~60-90 minutes (CPU) or ~15-20 minutes (GPU)

### 3. Test Model

```bash
python test_model.py
```

## 🌐 Deployment

### Deploy Backend

Options: Render, Railway, Heroku

```bash
cd backend
git push heroku main
```

### Deploy Frontend

Options: Vercel, Netlify

```bash
cd frontend
npm run build
# Deploy the build folder to Vercel/Netlify
```

After deployment, update API URL in `frontend/src/App.js`:

```javascript
const API_URL = "https://your-backend-url.com/api/predict";
```

## 📦 Dependencies

**Backend:**
- FastAPI
- TensorFlow/Keras
- Uvicorn
- Pillow
- NumPy

**Frontend:**
- React
- Axios

See `backend/requirements.txt` and `frontend/package.json` for full list.

## 🔗 Links

- **GitHub Repository:** [Your Repo URL]
- **Model Weights:** [Google Drive / Hugging Face Link]
- **Live Demo:** [Deployed URL - coming soon]

## 📝 Dataset

- **Dataset Name:** PlantVillage Dataset
- **Total Images:** 20,639
- **Classes:** 15
- **Source:** [PlantVillage](https://plantvillage.psu.edu)

## 🙏 Acknowledgments

- PlantVillage dataset
- TensorFlow/Keras community
- FastAPI framework
- React community

## 🚧 Future Improvements

- [ ] Add more crop diseases
- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time camera detection
- [ ] Disease recommendation system
- [ ] Multi-language support
- [ ] Offline mode

---

**Last Updated:** 2026