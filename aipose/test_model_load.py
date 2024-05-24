# test_model_load.py
import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)
print("Model loaded successfully!")
