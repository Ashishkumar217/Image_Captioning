import pickle

from utils.preprocess import (
    load_captions,
    preprocess_captions,
    create_caption_mapping
)

from utils.tokenizer_utils import load_tokenizer
from utils.data_generator import data_generator

# Load captions
df = load_captions("dataset/captions.txt")
df = preprocess_captions(df)

mapping = create_caption_mapping(df)

# Load tokenizer
tokenizer = load_tokenizer("models/tokenizer.pkl")

# Load features
with open("features/features.pkl", "rb") as file:
    features = pickle.load(file)

VOCAB_SIZE = len(tokenizer.word_index) + 1
MAX_LENGTH = max(len(c.split()) for c in df["caption_text"])

generator = data_generator(
    mapping,
    features,
    tokenizer,
    MAX_LENGTH,
    VOCAB_SIZE,
    batch_size=2
)

(X1, X2), y = next(generator)

print("Image Features Shape :", X1.shape)
print("Caption Shape        :", X2.shape)
print("Target Shape         :", y.shape)