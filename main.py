import os
import warnings
import time

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#Download the dataset
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

#open file and decode it
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
# print(f"Length of text: {len(text)} characters")

# print(text[:250])

vocab = sorted(set(text))
# print(f"{len(vocab)} unique characters")

example_texts = ["abcdefg", "xyz"]

# TODO 1
chars = tf.strings.unicode_split(example_texts, input_encoding="UTF-8")
# print(chars)

# Convert characters to ids
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
ids = ids_from_chars(chars)
# print(ids)``

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

chars = chars_from_ids(ids)
# print(chars)

# join the chars into strings
tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


