import os
import pickle
from tqdm import tqdm
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# -----------------------
# Paths
# -----------------------
IMAGE_FOLDER = "dataset/images"
FEATURE_FOLDER = "features"
FEATURE_FILE = os.path.join(FEATURE_FOLDER, "features.pkl")

# -----------------------
# Load VGG16 Model
# -----------------------
base_model = VGG16(weights="imagenet")

# Remove final classification layer
model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

print("✅ VGG16 Loaded Successfully")

# Create features folder
os.makedirs(FEATURE_FOLDER, exist_ok=True)

features = {}

# Get all image names
image_names = os.listdir(IMAGE_FOLDER)

print(f"Processing {len(image_names)} images...\n")

for image_name in tqdm(image_names):

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    try:
        # Load image
        image = load_img(image_path, target_size=(224, 224))

        # Convert to numpy array
        image = img_to_array(image)

        # Reshape
        image = np.expand_dims(image, axis=0)

        # Preprocess
        image = preprocess_input(image)

        # Extract feature
        feature = model.predict(image, verbose=0)

        image_id = image_name.split(".")[0]

        features[image_id] = feature

    except Exception as e:
        print(f"Skipped {image_name}: {e}")

# Save features
with open(FEATURE_FILE, "wb") as file:
    pickle.dump(features, file)

print("\n✅ Feature Extraction Completed")
print(f"Features Saved: {FEATURE_FILE}")
print(f"Total Features: {len(features)}")