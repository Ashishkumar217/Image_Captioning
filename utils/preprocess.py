import pandas as pd
import string

def load_captions(file_path):
    captions = pd.read_csv(file_path, sep="|")
    return captions


def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = " ".join(caption.split())
    return caption


def preprocess_captions(df):
    df["caption_text"] = df["caption_text"].apply(clean_caption)

    df["caption_text"] = df["caption_text"].apply(
        lambda x: "startseq " + x + " endseq"
    )

    return df

def create_caption_mapping(df):

    mapping = {}

    for _, row in df.iterrows():

        image_id = row["image_name"].split(".")[0]

        caption = row["caption_text"]

        if image_id not in mapping:
            mapping[image_id] = []

        mapping[image_id].append(caption)

    return mapping