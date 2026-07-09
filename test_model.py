from utils.model_builder import build_model

VOCAB_SIZE = 8832
MAX_LENGTH = 38

model = build_model(VOCAB_SIZE, MAX_LENGTH)

model.summary()