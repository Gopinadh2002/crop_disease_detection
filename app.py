import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto"
)

# -------------------------------
# Device
# -------------------------------
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    st.error(f"Error setting up device: {e}")
    device = torch.device("cpu") # Fallback to CPU

# -------------------------------
# Class Names
# -------------------------------
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]
num_classes = len(class_names)

# -------------------------------
# Load Trained Model (with caching)
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found. Please ensure it's in the correct directory.")
        return None

model = load_model()

# -------------------------------
# Image Transforms
# -------------------------------
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img, model, transform, class_names):
    if model is None:
        return "Error", 0.0
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    predicted_class = class_names[pred.item()]
    formatted_class = predicted_class.replace("__", " ").replace("_", " ")
    return formatted_class, conf.item()

# -------------------------------
# Streamlit UI
# -------------------------------

# --- Sidebar ---
with st.sidebar:
    st.title("🌿 PlantDoc AI")
    st.header("Upload Options")
    
    # File uploader (remains the same)
    uploaded_file = st.file_uploader(
        "1. Choose a leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a plant leaf for disease detection."
    )
    
    st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # --- NEW: Button-activated camera logic ---
    
    # Initialize session state for camera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    def activate_camera():
        # Callback function to set camera state to active
        st.session_state.camera_active = True

    # The button to activate the camera
    st.button("2. Take a Picture", on_click=activate_camera)

    camera_file = None
    # The camera input will only be displayed if the button has been clicked
    if st.session_state.camera_active:
        camera_file = st.camera_input(
            "Position the leaf and take a picture",
            help="Use your device's camera to capture a leaf image.",
            key="camera_input_widget"
        )
        # If a photo is taken, we can turn the camera state off
        if camera_file is not None:
            st.session_state.camera_active = False
            
    # --- END NEW LOGIC ---
    
    st.markdown("---")
    st.info(
        "**Tip:** For best results, use a clear image with the leaf against a plain background."
    )

# --- Main Page ---
st.title("🌱 Plant Disease Classifier")
st.write("Upload an image or take a picture using the options on the sidebar to identify potential diseases.")

# Determine which input to use
img_file = uploaded_file if uploaded_file is not None else camera_file

# Run prediction if an image is available
if img_file:
    img = Image.open(img_file).convert("RGB")
    
    # Display the image
    st.image(img, caption="Here's the leaf you provided.", use_container_width=True)

    # Perform prediction and show a spinner
    with st.spinner("Analyzing the leaf... 🧐"):
        label, confidence = predict(img, model, val_test_transforms, class_names)
    
    st.markdown("---")
    st.subheader("🔬 Analysis Result")

    if label == "Error":
        st.error("Could not perform prediction. Is the model loaded correctly?")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Disease", label)
        with col2:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        
        st.progress(confidence)
        st.success(f"The model is {confidence*100:.2f}% confident that the disease is **{label}**.")

else:
    # A welcoming message when no image is uploaded
    st.info("Please upload an image or take a picture using the options on the left sidebar.")