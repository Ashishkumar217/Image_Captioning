from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Add
)

from tensorflow.keras.models import Model


def build_model(vocab_size, max_length):

    # Image feature input (4096)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    # Caption input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merge both branches
    decoder1 = Add()([fe2, se3])

    decoder2 = Dense(256, activation="relu")(decoder1)

    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model