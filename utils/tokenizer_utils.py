import pickle
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(captions):
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer


def save_tokenizer(tokenizer, filename):
    with open(filename, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(filename):
    with open(filename, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer