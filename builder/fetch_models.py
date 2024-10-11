from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import os

model_names = ["base", "large-v3", "large-v3-turbo-ct2"]

def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(5):
        while True:
            try:
                if selected_model == "large-v3-turbo-ct2":
                    repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
                    local_dir = "faster-whisper-large-v3-turbo-ct2"
                    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")
                    loaded_model = WhisperModel(local_dir, device="cuda", compute_type="float16")
                else:
                    loaded_model = WhisperModel(
                        selected_model, device="cuda", compute_type="float16")
            except (AttributeError, OSError):
                continue

            break

    return selected_model, loaded_model

models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model