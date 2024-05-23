import tensorflow as tf
import tensorflow_hub as hub
import os

def download_and_save_model():
    try:
        # Load the model from TensorFlow Hub
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        
        # Define the path to save the model
        model_save_path = "./models/movenet_model"
        
        # Create the models directory if it does not exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Create a function to run the model and save it with serving signatures
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 192, 192, 3], dtype=tf.int32)])
        def serve_fn(inputs):
            return model.signatures['serving_default'](inputs)
        
        # Save the model with serving signatures
        tf.saved_model.save(model, model_save_path, signatures={'serving_default': serve_fn})
        print("Model downloaded and saved successfully with serving signatures.")
        
        # Verify the saved model directory
        if os.path.exists(model_save_path):
            print("Model directory exists.")
            print("Contents of the model directory:", os.listdir(model_save_path))
        else:
            print("Model directory does not exist.")
            
    except Exception as e:
        print("Failed to download and save the MoveNet model.", str(e))

if __name__ == "__main__":
    download_and_save_model()
