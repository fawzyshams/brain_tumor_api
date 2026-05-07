from fastapi import FastAPI, UploadFile, File
from typing import Annotated, List
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# ======================
# Config
# ======================
MODEL_PATH = "model.h5"

# تحميل الموديل لو مش موجود
if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/uc?id=1MydwW_aiPfoJOvU48DDioSHv3ghR_25h"

    gdown.download(url, MODEL_PATH, quiet=False)

# تحميل الموديل
model = tf.keras.models.load_model(MODEL_PATH)
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_IMAGES = 10

# ======================
# Load Model
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# Preprocessing Function
# ======================
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image = np.array(image)
        image = image / 255.0
        return image
    except Exception:
        return None


# ======================
# Home Route
# ======================
@app.get("/")
def home():
    return {
        "message": "AI Model API is running"
    }


# ======================
# Predict Endpoint
# ======================
@app.post("/predict")
async def predict(files: Annotated[List[UploadFile], File()]):

    # check max images
    if len(files) > MAX_IMAGES:
        return {"error": f"Maximum {MAX_IMAGES} images allowed"}

    processed_images = []

    for file in files:

        if not file.content_type.startswith("image/"):
            return {"error": f"{file.filename} is not an image"}

        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        if img is None:
            return {"error": f"Invalid image: {file.filename}"}

        processed_images.append(img)

    batch = np.array(processed_images)
    preds = model.predict(batch, verbose=0)
    avg_prediction = np.mean(preds, axis=0)
    class_index = int(np.argmax(avg_prediction))
    confidence = float(avg_prediction[class_index])

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }