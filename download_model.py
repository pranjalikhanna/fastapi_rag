import os
from huggingface_hub import hf_hub_download

def download_model():
    model_path = "models/llama-2-7b.ggmlv3.q4_0.bin"
    if not os.path.exists(model_path):
        print("Downloading model...")
        hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGML", filename="llama-2-7b.ggmlv3.q4_0.bin", local_dir="models")
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()

