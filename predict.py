import pickle
import numpy as np

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from utils.tokenizer_utils import load_tokenizer

# -----------------------
# Load tokenizer and model
# -----------------------
tokenizer = load_tokenizer("models/tokenizer.pkl")
model = load_model("models/best_model.keras")

MAX_LENGTH = 38

# -----------------------
# Build VGG16 Feature Extractor
# -----------------------
base_model = VGG16(weights="imagenet")
feature_extractor = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)


def extract_feature(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)

    return feature


def index_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


def generate_caption(model, tokenizer, feature, max_length):

    caption = "startseq"

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences([caption])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        prediction = model.predict([feature, sequence], verbose=0)

        predicted_index = np.argmax(prediction)

        word = index_to_word(predicted_index, tokenizer)

        if word is None:
            break

        caption += " " + word

        if word == "endseq":
            break

    return caption.replace("startseq", "").replace("endseq", "").strip()


# -----------------------
# Test Image
# -----------------------
image_path = "dataset/images/1000268201_693b08cb0e.jpg"

feature = extract_feature(image_path)

caption = generate_caption(
    model,
    tokenizer,
    feature,
    MAX_LENGTH
)

print("\nGenerated Caption:")
print(caption)