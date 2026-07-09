import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)

from utils.tokenizer_utils import load_tokenizer

# -------------------------------
# Load tokenizer and model
# -------------------------------
tokenizer = load_tokenizer("models/tokenizer.pkl")
model = load_model("models/best_model.keras")

MAX_LENGTH = 38

# -------------------------------
# Load VGG16
# -------------------------------
base_model = VGG16(weights="imagenet")

feature_extractor = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_feature(image):

    image = image.resize((224,224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)

    return feature

# -------------------------------
# Convert index to word
# -------------------------------
def index_to_word(index):

    for word, idx in tokenizer.word_index.items():

        if idx == index:
            return word

    return None

# -------------------------------
# Caption Generator
# -------------------------------
def generate_caption(feature):

    caption = "startseq"

    for _ in range(MAX_LENGTH):

        sequence = tokenizer.texts_to_sequences([caption])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=MAX_LENGTH,
            padding="post"
        )

        prediction = model.predict(
            [feature, sequence],
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        word = index_to_word(predicted_index)

        if word is None:
            break

        caption += " " + word

        if word == "endseq":
            break

    caption = caption.replace("startseq","")
    caption = caption.replace("endseq","")

    return caption.strip()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Image Captioning AI",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Captioning AI")

st.write(
    "Upload an image and let the AI generate a caption."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating Caption..."):

            feature = extract_feature(image)

            caption = generate_caption(feature)

        st.success("Caption Generated!")

        st.subheader("Generated Caption")

        st.write(caption)