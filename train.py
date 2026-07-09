import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from utils.preprocess import (
    load_captions,
    preprocess_captions,
    create_caption_mapping,
)

from utils.tokenizer_utils import load_tokenizer
from utils.model_builder import build_model
from utils.data_generator import data_generator

# -----------------------
# Load captions
# -----------------------
df = load_captions("dataset/captions.txt")
df = preprocess_captions(df)

mapping = create_caption_mapping(df)

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = load_tokenizer("models/tokenizer.pkl")

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in df["caption_text"])

# -----------------------
# Load image features
# -----------------------
with open("features/features.pkl", "rb") as file:
    features = pickle.load(file)

# -----------------------
# Train / Validation split
# -----------------------
image_ids = list(mapping.keys())

train_ids, val_ids = train_test_split(
    image_ids,
    test_size=0.2,
    random_state=42
)

train_mapping = {k: mapping[k] for k in train_ids}
val_mapping = {k: mapping[k] for k in val_ids}

# -----------------------
# Generators
# -----------------------
batch_size = 32

train_generator = data_generator(
    train_mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
)

val_generator = data_generator(
    val_mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
)

steps_per_epoch = len(train_mapping) // batch_size
validation_steps = len(val_mapping) // batch_size

# -----------------------
# Build model
# -----------------------
model = build_model(vocab_size, max_length)

# -----------------------
# Callbacks
# -----------------------
checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------
# Train
# -----------------------
history = model.fit(
    train_generator,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("models/image_caption_model.keras")

print("✅ Training Complete!")