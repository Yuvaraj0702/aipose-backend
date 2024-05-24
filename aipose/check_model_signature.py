import tensorflow as tf
import os

def check_model_signature(model_path="../models/movenet_model"):
    # Verify if the model directory exists and contains the expected files
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path '{model_path}' does not exist.")
    if not any(os.scandir(model_path)):
        raise RuntimeError(f"Model path '{model_path}' is empty.")
    print("Model path exists and is not empty.")
    print("Contents of the model path:", os.listdir(model_path))

    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully.")
        print("Available signatures: ", list(model.signatures.keys()))
    except Exception as e:
        raise RuntimeError("Failed to load the MoveNet model from the specified path.") from e

if __name__ == "__main__":
    check_model_signature()
