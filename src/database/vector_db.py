import os
import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Classe para gerenciar o banco de dados vetorial usando ChromaDB."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Inicializa a conexão com o banco de dados vetorial.
        
        Args:
            persist_directory: Diretório onde os dados serão persistidos.
            collection_name: Nome da coleção para armazenar embeddings.
        """
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
        except Exception as e:
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
        """
        Adiciona itens à coleção vetorial.
        
        Args:
            ids: Lista de IDs únicos para os itens.
            embeddings: Lista de embeddings. Se None, serão gerados a partir dos documentos.
            documents: Lista de textos descritivos para os itens.
            metadatas: Lista de metadados para os itens.
            
        Returns:
            bool: True se a operação foi bem-sucedida.
        """
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
                limit: int = 10,
                similarity_threshold: float = 0.70,
                feature_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Realiza uma consulta no banco vetorial com técnicas avançadas para aumentar similaridade.
        
        Args:
            query_text: Texto da consulta para gerar embedding.
            query_embedding: Embedding pré-calculado para a consulta.
            filter_criteria: Critérios de filtro para a consulta.
            limit: Número máximo de resultados iniciais para permitir reranking.
            similarity_threshold: Limiar mínimo de similaridade (0-1).
            feature_weights: Pesos para diferentes características da consulta.
            
        Returns:
            Dict: Resultados da consulta após reranking.
        """
        
        logger.info(f"Realizando consulta vetorial. Embedding shape: {len(query_embedding)}")
        logger.info(f"Limite de resultados: {limit}")
        if not query_text and not query_embedding:
            raise ValueError("É necessário fornecer query_text ou query_embedding")
            
        try:
            if feature_weights is None:
                feature_weights = {
                    "tipo_peca": 2.0,
                    "cor": 1.5,
                    "padrao": 1.2,
                    "estilo": 1.8,
                    "ocasiao": 1.5,
                    "genero": 1.3,
                    "estacao": 1.0,
                    "material": 1.0
                }
            
            if query_embedding:
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = [float(val/norm) for val in query_embedding]
            
            initial_results = self.collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=[query_embedding] if query_embedding else None,
                where=filter_criteria,
                n_results=min(limit * 3, 30),
                include=["metadatas", "documents", "distances", "embeddings"]
            )
            
            if len(initial_results["ids"][0]) > 0:
                enhanced_scores = []
                
                for i in range(len(initial_results["ids"][0])):
                    base_similarity = 1 - initial_results["distances"][0][i]
                    
                    document = initial_results["documents"][0][i] if initial_results["documents"] else ""
                    metadata = initial_results["metadatas"][0][i] if initial_results["metadatas"] else {}
                    
                    weighted_score = base_similarity

                    try:
                        import json
                        import re
                        
                        json_match = re.search(r'\{.*\}', document, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group(0))
                                
                                if query_text:
                                    if "tipo_peca" in json_data and json_data["tipo_peca"].lower() in query_text.lower():
                                        weighted_score *= feature_weights.get("tipo_peca", 2.0)
                                    
                                    if "cores_predominantes" in json_data:
                                        cores = json_data["cores_predominantes"]
                                        if isinstance(cores, str):
                                            cores = [cores]
                                        for cor in cores:
                                            if cor.lower() in query_text.lower():
                                                weighted_score *= feature_weights.get("cor", 1.5)
                                                break
                                    
                                    for field, weight_key in [
                                        ("padrao", "padrao"),
                                        ("estilo", "estilo"),
                                        ("ocasiao_uso", "ocasiao"),
                                        ("genero", "genero"),
                                        ("estacao", "estacao"),
                                        ("material", "material")
                                    ]:
                                        if field in json_data and isinstance(json_data[field], str) and json_data[field].lower() in query_text.lower():
                                            weighted_score *= feature_weights.get(weight_key, 1.0)
                            except json.JSONDecodeError:
                                pass
                        
                        if query_text and not json_match:
                            keywords = ["vestido", "camisa", "calça", "casaco", "jaqueta", "saia", "blusa",  # tipos
                                     "vermelho", "azul", "verde", "preto", "branco", "amarelo", "rosa",    # cores
                                     "formal", "casual", "esportivo", "elegante", "vintage", "moderno",     # estilos
                                     "verão", "inverno", "outono", "primavera",                             # estações
                                     "floral", "listrado", "xadrez", "liso", "estampado"]                   # padrões
                            
                            for keyword in keywords:
                                if keyword in query_text.lower() and keyword in document.lower():
                                    if keyword in ["vestido", "camisa", "calça", "casaco", "jaqueta", "saia", "blusa"]:
                                        weighted_score *= feature_weights.get("tipo_peca", 2.0)
                                    elif keyword in ["vermelho", "azul", "verde", "preto", "branco", "amarelo", "rosa"]:
                                        weighted_score *= feature_weights.get("cor", 1.5)
                                    elif keyword in ["formal", "casual", "esportivo", "elegante", "vintage", "moderno"]:
                                        weighted_score *= feature_weights.get("estilo", 1.8)
                                    elif keyword in ["verão", "inverno", "outono", "primavera"]:
                                        weighted_score *= feature_weights.get("estacao", 1.0)
                                    elif keyword in ["floral", "listrado", "xadrez", "liso", "estampado"]:
                                        weighted_score *= feature_weights.get("padrao", 1.2)
                    except Exception as e:
                        logger.debug(f"Erro ao processar ponderação: {str(e)}")
                    
                    filename = metadata.get("filename", "")
                    if query_text and filename:
                        clean_filename = filename.replace("_220x220", "").replace(".jpg", "").replace("-", " ").replace("_", " ").lower()
                        
                        for word in query_text.lower().split():
                            if len(word) > 3 and word in clean_filename:
                                weighted_score *= 1.3
                    
                    if weighted_score > 1.0:
                        normalized_score = 1.0 / (1.0 + np.exp(-(weighted_score - 1.0) * 2))
                        normalized_score = base_similarity + (1.0 - base_similarity) * normalized_score
                    else:
                        normalized_score = weighted_score
                    
                    final_score = max(min(normalized_score, 1.0), base_similarity)
                    
                    if final_score > 0.85:
                        amplification_factor = 0.7
                        final_score = final_score + (1.0 - final_score) * amplification_factor
                    
                    enhanced_scores.append((i, final_score))
                
                enhanced_scores.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in enhanced_scores[:limit]]
                top_scores = [score for _, score in enhanced_scores[:limit]]
                
                reranked_results = {
                    "ids": [[initial_results["ids"][0][i] for i in top_indices]],
                    "metadatas": [[initial_results["metadatas"][0][i] if initial_results["metadatas"] else None for i in top_indices]],
                    "documents": [[initial_results["documents"][0][i] if initial_results["documents"] else None for i in top_indices]],
                    "distances": [[1.0 - score for score in top_scores]]
                }
                
                return reranked_results
            
            return initial_results
            
        except Exception as e:
            logger.error(f"Erro na consulta: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtém informações sobre a coleção.
        
        Returns:
            Dict: Informações da coleção.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "item_count": count,
            "persist_directory": self.persist_directory
        }
        
    def delete_collection(self) -> bool:
        """
        Remove a coleção atual.
        
        Returns:
            bool: True se a operação foi bem-sucedida.
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Coleção '{self.collection_name}' removida com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao remover coleção: {str(e)}")
            return False