from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = FastAPI(
    title="Flagship Wheat Disease Detection API",
    description="High-performance inference API for wheat disease classification.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Variable
model = None
CLASSES = ["Crown and Root Rot", "Healthy Wheat", "Leaf Rust", "Wheat Loose Smut"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/flagship_model.keras")

@app.on_event("startup")
async def load_ml_model():
    global model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"WARNING: Model file not found at {MODEL_PATH}. API will return mock predictions.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")

@app.get("/")
def read_root():
    return {"message": "Wheat Analysis Flagship API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    img_np = np.array(image)
    # Resize to 260x260 (EfficientNetV2B2 native)
    img_resized = cv2.resize(img_np, (260, 260))
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # Inference
    if model:
        predictions = model.predict(img_batch)
        confidence_scores = predictions[0]
        predicted_class_index = np.argmax(confidence_scores)
        raw_confidence = float(confidence_scores[predicted_class_index])
        
        # PROBABILITY THRESHOLD: Filter out non-wheat images
        # If the model isn't at least 70% sure, we flag it.
        THRESHOLD = 0.70
        
        scores_dict = {cls: float(score) for cls, score in zip(CLASSES, confidence_scores)}
        
        if raw_confidence < THRESHOLD:
            result = {
                "prediction": "Unknown / Not Wheat",
                "confidence": raw_confidence,
                "scores": scores_dict,
                "alert": "Low confidence. Please ensure the image is a clear leaf photo."
            }
        else:
            result = {
                "prediction": CLASSES[predicted_class_index],
                "confidence": raw_confidence,
                "scores": scores_dict
            }
    else:
        # Mock Response for Dev/Testing if model isn't trained yet
        import random
        mock_class = random.choice(CLASSES)
        result = {
            "prediction": mock_class,
            "confidence": 0.95,
            "scores": {c: (0.95 if c == mock_class else 0.01) for c in CLASSES},
            "warning": "Mock prediction (Model not loaded)"
        }
        
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
