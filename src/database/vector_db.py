import os
import logging
from typing import Dict, List, Any, Optional, Union

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):

        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "fashion_embeddings")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL")
        )
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Coleção '{self.collection_name}' conectada com sucesso.")
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Coleção '{self.collection_name}' criada com sucesso.")
    
    def add_items(self, 
                  ids: List[str], 
                  embeddings: Optional[List[List[float]]] = None,
                  documents: Optional[List[str]] = None,
                  metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Adicionados {len(ids)} itens à coleção '{self.collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar itens: {str(e)}")
            return False
    
    def query(self, 
              query_text: Optional[str] = None,
              query_embedding: Optional[List[float]] = None,
              filter_criteria: Optional[Dict[str, Any]] = None,
              limit: int = 5) -> Dict[str, Any]:

        if not query_text and not query_embedding:
            raise ValueError("É necessário fornecer query_text ou query_embedding")
            
        try:
            results = self.collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=[query_embedding] if query_embedding else None,
                where=filter_criteria,
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            return results
        except Exception as e:
            logger.error(f"Erro na consulta: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:

        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "item_count": count,
            "persist_directory": self.persist_directory
        }
        
    def delete_collection(self) -> bool:

        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Coleção '{self.collection_name}' removida com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao remover coleção: {str(e)}")
            return False