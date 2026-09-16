"""Download the local Generator & Refiner model weights (GGUF, Q4_K_M) from
Hugging Face into models/. One-time ~9GB download; re-running is a no-op if
the file already exists (huggingface_hub resumes/skips via local_dir caching).

Usage: python scripts/download_local_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from papers.execution_aware_alpha_mining.src import config

if __name__ == "__main__":
    path = hf_hub_download(
        repo_id=config.LOCAL_MODEL_REPO,
        filename=config.LOCAL_MODEL_FILE,
        local_dir=str(config.MODELS_DIR),
    )
    print(f"Model ready at: {path}")
