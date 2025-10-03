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

