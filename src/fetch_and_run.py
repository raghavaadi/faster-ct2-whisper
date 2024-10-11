import os
from faster_whisper import WhisperModel

# Function to download the model and use FP16 and CUDA
def download_models():
    model_names = ["tiny", "base", "large-v2"]  # Update this list if needed
    models = {}
    for model_name in model_names:
        print(f"Downloading model: {model_name}")
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        models[model_name] = model
    print("Models downloaded successfully!")
    return models

if __name__ == "__main__":
    # Step 1: Download models
    models = download_models()
    
    # Step 2: Run your main script (e.g., rp_handler.py)
    os.system("python -u /rp_handler.py")
