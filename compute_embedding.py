# compute_embeddings.py
import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Config
IMAGE_DIR = "images"
IMAGE_SIZE = (224, 224)   # same size you used in Colab
OUTPUT_EMB = "resnet_features.pkl"
OUTPUT_FILES = "resnet_filenames.pkl"


# Load ResNet50 (no top, global avg pooling)
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

features = []
filenames = []

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(IMAGE_DIR, fname)
    try:
        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        arr = np.array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0)           # (1, H, W, 3)
        feat = model.predict(arr, verbose=0)  # (1, D)
        features.append(feat.flatten())
        filenames.append(path)
        print("Processed:", fname)
    except Exception as e:
        print("Skipped:", fname, "->", e)

features = np.array(features)

with open(OUTPUT_EMB, "wb") as f:
    pickle.dump(features, f)

with open(OUTPUT_FILES, "wb") as f:
    pickle.dump(filenames, f)

print(f"Done — saved {OUTPUT_EMB} and {OUTPUT_FILES} for {len(filenames)} images.")
