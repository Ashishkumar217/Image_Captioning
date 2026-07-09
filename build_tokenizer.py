from utils.preprocess import load_captions, preprocess_captions
from utils.tokenizer_utils import create_tokenizer, save_tokenizer

# Load and preprocess captions
captions_df = load_captions("dataset/captions.txt")
captions_df = preprocess_captions(captions_df)

captions = captions_df["caption_text"].tolist()

# Create tokenizer
tokenizer = create_tokenizer(captions)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Maximum caption length
max_length = max(len(caption.split()) for caption in captions)

print("Vocabulary Size:", vocab_size)
print("Maximum Caption Length:", max_length)

# Save tokenizer
save_tokenizer(tokenizer, "models/tokenizer.pkl")

print("Tokenizer saved successfully!")