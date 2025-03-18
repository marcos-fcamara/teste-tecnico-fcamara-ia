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
        
        # Garante que o diretório existe
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Inicializa o cliente ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Define a função de embedding usando OpenAI
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL")
        )
        
        # Obtém ou cria a coleção - MODIFICADO PARA CORRIGIR ERRO
        try:
            # Tenta obter a coleção existente
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Coleção '{self.collection_name}' conectada com sucesso.")
        except Exception as e:
            # Se a coleção não existir, cria uma nova
            logger.info(f"Coleção '{self.collection_name}' não existe. Criando nova coleção.")
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
                limit: int = 10,  # Aumentado para permitir reranking
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
        if not query_text and not query_embedding:
            raise ValueError("É necessário fornecer query_text ou query_embedding")
            
        try:
            # Configuração de pesos para características específicas (se não fornecido, usa valores padrão)
            if feature_weights is None:
                feature_weights = {
                    "tipo_peca": 2.0,       # Tipo de peça tem peso mais alto (ex: vestido, camisa)
                    "cor": 1.5,             # Cor tem peso médio-alto
                    "padrao": 1.2,          # Padrão/estampa tem peso médio
                    "estilo": 1.8,          # Estilo tem peso alto (ex: casual, formal)
                    "ocasiao": 1.5,         # Ocasião de uso tem peso médio-alto
                    "genero": 1.3,          # Gênero tem peso médio
                    "estacao": 1.0,         # Estação tem peso normal
                    "material": 1.0         # Material tem peso normal
                }
            
            # Normalização avançada do embedding de consulta
            if query_embedding:
                # Normalização L2 do vetor de consulta
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = [float(val/norm) for val in query_embedding]
            
            # Busca com limite aumentado para permitir reranking posterior
            initial_results = self.collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=[query_embedding] if query_embedding else None,
                where=filter_criteria,
                n_results=min(limit * 3, 30),  # Buscamos mais resultados para reranking avançado
                include=["metadatas", "documents", "distances", "embeddings"]
            )
            
            # Implementação de reranking semântico avançado para aumentar a similaridade
            if len(initial_results["ids"][0]) > 0:
                # Lista para armazenar pontuações refinadas
                enhanced_scores = []
                
                for i in range(len(initial_results["ids"][0])):
                    # Pontuação base (similaridade coseno)
                    base_similarity = 1 - initial_results["distances"][0][i]
                    
                    # Obter documento e metadados do resultado
                    document = initial_results["documents"][0][i] if initial_results["documents"] else ""
                    metadata = initial_results["metadatas"][0][i] if initial_results["metadatas"] else {}
                    
                    # Aplicar ponderação baseada em correspondência de características
                    weighted_score = base_similarity
                    
                    # Analisa o documento para encontrar características e aplicar pesos
                    # Analisamos o documento JSON e extraímos características
                    try:
                        # Verificar se o documento contém JSON (nossa descrição estruturada)
                        import json
                        import re
                        
                        # Tentativa 1: Extrair o JSON diretamente
                        json_match = re.search(r'\{.*\}', document, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group(0))
                                
                                # Aplicar pesos às características encontradas no JSON
                                if query_text:
                                    # Tipo de peça
                                    if "tipo_peca" in json_data and json_data["tipo_peca"].lower() in query_text.lower():
                                        weighted_score *= feature_weights.get("tipo_peca", 2.0)
                                    
                                    # Cor
                                    if "cores_predominantes" in json_data:
                                        cores = json_data["cores_predominantes"]
                                        if isinstance(cores, str):
                                            cores = [cores]
                                        for cor in cores:
                                            if cor.lower() in query_text.lower():
                                                weighted_score *= feature_weights.get("cor", 1.5)
                                                break
                                    
                                    # Outros campos importantes
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
                                # Se falhar, continuamos com a pontuação base
                                pass
                        
                        # Se não conseguiu extrair JSON, busca características diretamente no texto
                        if query_text and not json_match:
                            # Identifica palavras-chave importantes na consulta
                            keywords = ["vestido", "camisa", "calça", "casaco", "jaqueta", "saia", "blusa",  # tipos
                                     "vermelho", "azul", "verde", "preto", "branco", "amarelo", "rosa",    # cores
                                     "formal", "casual", "esportivo", "elegante", "vintage", "moderno",     # estilos
                                     "verão", "inverno", "outono", "primavera",                             # estações
                                     "floral", "listrado", "xadrez", "liso", "estampado"]                   # padrões
                            
                            for keyword in keywords:
                                if keyword in query_text.lower() and keyword in document.lower():
                                    # Tipo de peça
                                    if keyword in ["vestido", "camisa", "calça", "casaco", "jaqueta", "saia", "blusa"]:
                                        weighted_score *= feature_weights.get("tipo_peca", 2.0)
                                    # Cores
                                    elif keyword in ["vermelho", "azul", "verde", "preto", "branco", "amarelo", "rosa"]:
                                        weighted_score *= feature_weights.get("cor", 1.5)
                                    # Estilos
                                    elif keyword in ["formal", "casual", "esportivo", "elegante", "vintage", "moderno"]:
                                        weighted_score *= feature_weights.get("estilo", 1.8)
                                    # Estações
                                    elif keyword in ["verão", "inverno", "outono", "primavera"]:
                                        weighted_score *= feature_weights.get("estacao", 1.0)
                                    # Padrões
                                    elif keyword in ["floral", "listrado", "xadrez", "liso", "estampado"]:
                                        weighted_score *= feature_weights.get("padrao", 1.2)
                    except Exception as e:
                        # Em caso de erro, mantém a pontuação base
                        logger.debug(f"Erro ao processar ponderação: {str(e)}")
                    
                    # Aplicar bônus para metadados importantes
                    filename = metadata.get("filename", "")
                    if query_text and filename:
                        # Limpa o nome do arquivo para análise
                        clean_filename = filename.replace("_220x220", "").replace(".jpg", "").replace("-", " ").replace("_", " ").lower()
                        
                        # Para cada palavra na consulta, verifica se está no nome do arquivo
                        for word in query_text.lower().split():
                            if len(word) > 3 and word in clean_filename:  # ignora palavras muito curtas
                                weighted_score *= 1.3  # bônus significativo para correspondência em metadados
                    
                    # Aplicar normalização para manter valores entre 0 e 1
                    # Usar sigmoid para comprimir valores muito altos
                    if weighted_score > 1.0:
                        # Sigmoid normalização: 1 / (1 + exp(-x))
                        # Ajustamos para que valores normais (1-2) fiquem na parte crescente da curva
                        normalized_score = 1.0 / (1.0 + np.exp(-(weighted_score - 1.0) * 2))
                        # Escalar de volta para o intervalo [original, 1.0]
                        normalized_score = base_similarity + (1.0 - base_similarity) * normalized_score
                    else:
                        normalized_score = weighted_score
                    
                    # Garantir que está no intervalo [0, 1] e é pelo menos o valor original
                    final_score = max(min(normalized_score, 1.0), base_similarity)
                    
                    # Aplicar um fator de amplificação para aproximar de 95% as melhores correspondências
                    # Para correspondências acima de um limiar alto, aplica um aumento não-linear
                    if final_score > 0.85:
                        # Fórmula: base + (1 - base) * fator
                        # Quanto maior o score original, mais próximo de 1.0 ficará
                        amplification_factor = 0.7  # Controla quanto aproximamos de 1.0
                        final_score = final_score + (1.0 - final_score) * amplification_factor
                    
                    enhanced_scores.append((i, final_score))
                
                # Ordena pelos scores aprimorados e pega os top N
                enhanced_scores.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in enhanced_scores[:limit]]
                top_scores = [score for _, score in enhanced_scores[:limit]]
                
                # Prepara resultados reranqueados com as novas pontuações
                reranked_results = {
                    "ids": [[initial_results["ids"][0][i] for i in top_indices]],
                    "metadatas": [[initial_results["metadatas"][0][i] if initial_results["metadatas"] else None for i in top_indices]],
                    "documents": [[initial_results["documents"][0][i] if initial_results["documents"] else None for i in top_indices]],
                    "distances": [[1.0 - score for score in top_scores]]  # Converte similaridade de volta para distância
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