import os
import pandas as pd

# Paths
IMAGE_PATH = "dataset/images"
CAPTION_PATH = "dataset/captions.txt"

# Check images
images = os.listdir(IMAGE_PATH)

print(f"Total Images: {len(images)}")
print("First 5 Images:")
print(images[:5])

# Read captions
captions = pd.read_csv(CAPTION_PATH, sep="|")

print("\nDataset Shape:")
print(captions.shape)

print("\nColumn Names:")
print(captions.columns)

print("\nFirst 5 Rows:")
print(captions.head())