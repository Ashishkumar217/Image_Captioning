from utils.preprocess import load_captions, preprocess_captions

captions = load_captions("dataset/captions.txt")

print("Before Cleaning:\n")
print(captions["caption_text"].head())

captions = preprocess_captions(captions)

print("\nAfter Cleaning:\n")
print(captions["caption_text"].head())