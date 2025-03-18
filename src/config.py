import os
import logging
from dotenv import load_dotenv
import json

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for directory in [DATA_DIR, IMAGES_DIR, CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", os.path.join(BASE_DIR, "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fashion_embeddings")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY não definida. A aplicação não funcionará corretamente.")

def get_config():
    return {
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "IMAGES_DIR": IMAGES_DIR,
        "CACHE_DIR": CACHE_DIR,
        "LOG_DIR": LOG_DIR,
        "OPENAI_API_KEY": "***" if OPENAI_API_KEY else None,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "VISION_MODEL": VISION_MODEL,
        "CHROMA_PERSIST_DIRECTORY": CHROMA_PERSIST_DIRECTORY,
        "CHROMA_COLLECTION_NAME": CHROMA_COLLECTION_NAME,
        "API_HOST": API_HOST,
        "API_PORT": API_PORT,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_WORKERS": MAX_WORKERS
    }

if __name__ == "__main__":
    config = get_config()
    print(json.dumps(config, indent=2))