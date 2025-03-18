#!/usr/bin/env python3
"""
Script para configurar o ambiente do projeto.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Cria a estrutura de diretórios necessária para o projeto."""
    directories = [
        "src/data",
        "src/data/images",
        "src/data/processed",
        "src/data/cache",
        "src/data/cache/descriptions",
        "src/data/cache/embeddings",
        "src/data/cache/queries",
        
        "chroma_db",
        
        "logs",
        
        "tests_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório criado/verificado: {directory}")

def create_env_file():
    """Cria um arquivo .env de exemplo se ele não existir."""
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("""# OpenAI
OPENAI_API_KEY=

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=fashion_embeddings

# API
API_HOST=0.0.0.0
API_PORT=7071
ALLOWED_ORIGINS=*

# Modelos
EMBEDDING_MODEL=text-embedding-3-small
VISION_MODEL=gpt-4o

# Processamento
BATCH_SIZE=10
MAX_WORKERS=4
SIMILARITY_THRESHOLD=0.7
TOP_K_RESULTS=5
""")
        logger.info(f"Arquivo .env de exemplo criado.")
    else:
        logger.info(f"Arquivo .env já existe.")

def main():
    """Função principal."""
    logger.info("Configurando ambiente do projeto...")
    
    try:
        create_directories()
        create_env_file()
        logger.info("Configuração concluída com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante a configuração: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()