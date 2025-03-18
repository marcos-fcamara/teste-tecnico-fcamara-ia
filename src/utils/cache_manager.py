"""
Gerenciador de cache para o sistema de busca por similaridade.
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import pickle
import time
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Gerenciador de cache para resultados de processamento de imagens e consultas.
    """
    
    def __init__(self, cache_dir: str = "src/data/cache"):
        """
        Inicializa o gerenciador de cache.
        
        Args:
            cache_dir: Diretório para armazenar os arquivos de cache.
        """
        self.cache_dir = cache_dir
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.descriptions_dir = os.path.join(cache_dir, "descriptions")
        self.queries_dir = os.path.join(cache_dir, "queries")
        
        # Cria os diretórios de cache
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.descriptions_dir, exist_ok=True)
        os.makedirs(self.queries_dir, exist_ok=True)
        
        # Contadores para estatísticas
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _generate_key(self, data: Union[str, bytes, Dict]) -> str:
        """
        Gera uma chave de cache a partir dos dados fornecidos.
        
        Args:
            data: Dados para gerar a chave.
            
        Returns:
            str: Chave de cache.
        """
        if isinstance(data, Dict):
            data = json.dumps(data, sort_keys=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return hashlib.md5(data).hexdigest()
    
    def _get_file_path(self, key: str, cache_type: str) -> str:
        """
        Obtém o caminho completo para um arquivo de cache.
        
        Args:
            key: Chave do cache.
            cache_type: Tipo de cache (embeddings, descriptions, queries).
            
        Returns:
            str: Caminho completo para o arquivo de cache.
        """
        if cache_type == "embeddings":
            return os.path.join(self.embeddings_dir, f"{key}.pkl")
        elif cache_type == "descriptions":
            return os.path.join(self.descriptions_dir, f"{key}.json")
        elif cache_type == "queries":
            return os.path.join(self.queries_dir, f"{key}.pkl")
        else:
            return os.path.join(self.cache_dir, f"{cache_type}_{key}.pkl")
    
    def get_cached_description(self, image_path: str) -> Optional[str]:
        """
        Obtém a descrição em cache para uma imagem.
        
        Args:
            image_path: Caminho da imagem.
            
        Returns:
            Optional[str]: Descrição em cache ou None se não encontrada.
        """
        # Gera a chave a partir do caminho e do timestamp de modificação
        mtime = os.path.getmtime(image_path)
        key = self._generate_key(f"{image_path}_{mtime}")
        file_path = self._get_file_path(key, "descriptions")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_hits += 1
                    logger.debug(f"Cache hit para descrição: {image_path}")
                    return data["description"]
            except Exception as e:
                logger.warning(f"Erro ao ler cache de descrição: {str(e)}")
        
        self.cache_misses += 1
        logger.debug(f"Cache miss para descrição: {image_path}")
        return None
    
    def cache_description(self, image_path: str, description: str) -> bool:
        """
        Armazena a descrição de uma imagem em cache.
        
        Args:
            image_path: Caminho da imagem.
            description: Descrição da imagem.
            
        Returns:
            bool: True se o cache foi bem-sucedido.
        """
        # Gera a chave a partir do caminho e do timestamp de modificação
        mtime = os.path.getmtime(image_path)
        key = self._generate_key(f"{image_path}_{mtime}")
        file_path = self._get_file_path(key, "descriptions")
        
        try:
            data = {
                "image_path": image_path,
                "timestamp": time.time(),
                "mtime": mtime,
                "description": description
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Descrição armazenada em cache: {image_path}")
            return True
        except Exception as e:
            logger.warning(f"Erro ao armazenar descrição em cache: {str(e)}")
            return False
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Obtém o embedding em cache para um texto.
        
        Args:
            text: Texto de entrada.
            
        Returns:
            Optional[List[float]]: Embedding em cache ou None se não encontrado.
        """
        key = self._generate_key(text)
        file_path = self._get_file_path(key, "embeddings")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_hits += 1
                    logger.debug(f"Cache hit para embedding: {text[:30]}...")
                    return data["embedding"]
            except Exception as e:
                logger.warning(f"Erro ao ler cache de embedding: {str(e)}")
        
        self.cache_misses += 1
        logger.debug(f"Cache miss para embedding: {text[:30]}...")
        return None
    
    def cache_embedding(self, text: str, embedding: List[float]) -> bool:
        """
        Armazena o embedding de um texto em cache.
        
        Args:
            text: Texto de entrada.
            embedding: Vetor de embedding.
            
        Returns:
            bool: True se o cache foi bem-sucedido.
        """
        key = self._generate_key(text)
        file_path = self._get_file_path(key, "embeddings")
        
        try:
            data = {
                "text": text,
                "timestamp": time.time(),
                "embedding": embedding
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug(f"Embedding armazenado em cache: {text[:30]}...")
            return True
        except Exception as e:
            logger.warning(f"Erro ao armazenar embedding em cache: {str(e)}")
            return False
    
    def get_cached_query_results(self, query: str, limit: int) -> Optional[Dict[str, Any]]:
        """
        Obtém os resultados em cache para uma consulta.
        
        Args:
            query: Texto da consulta.
            limit: Número máximo de resultados.
            
        Returns:
            Optional[Dict]: Resultados em cache ou None se não encontrados.
        """
        key = self._generate_key(f"{query}_{limit}")
        file_path = self._get_file_path(key, "queries")
        
        # Verifica se o cache existe e não está expirado (1 hora)
        if os.path.exists(file_path) and (time.time() - os.path.getmtime(file_path)) < 3600:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_hits += 1
                    logger.debug(f"Cache hit para consulta: {query}")
                    return data
            except Exception as e:
                logger.warning(f"Erro ao ler cache de consulta: {str(e)}")
        
        self.cache_misses += 1
        logger.debug(f"Cache miss para consulta: {query}")
        return None
    
    def cache_query_results(self, query: str, limit: int, results: Dict[str, Any]) -> bool:
        """
        Armazena os resultados de uma consulta em cache.
        
        Args:
            query: Texto da consulta.
            limit: Número máximo de resultados.
            results: Resultados da consulta.
            
        Returns:
            bool: True se o cache foi bem-sucedido.
        """
        key = self._generate_key(f"{query}_{limit}")
        file_path = self._get_file_path(key, "queries")
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
                
            logger.debug(f"Resultados de consulta armazenados em cache: {query}")
            return True
        except Exception as e:
            logger.warning(f"Erro ao armazenar resultados de consulta em cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Obtém estatísticas do uso do cache.
        
        Returns:
            Dict: Estatísticas do cache.
        """
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": self.cache_hits + self.cache_misses,
            "hit_ratio": self.cache_hits / max(1, (self.cache_hits + self.cache_misses))
        }
    
    def clear_cache(self, cache_type: Optional[str] = None) -> int:
        """
        Limpa os arquivos de cache.
        
        Args:
            cache_type: Tipo de cache a ser limpo. Se None, limpa todos os tipos.
            
        Returns:
            int: Número de arquivos removidos.
        """
        removed = 0
        
        if cache_type == "descriptions" or cache_type is None:
            removed += sum(1 for _ in Path(self.descriptions_dir).glob("*.json") if _.unlink() or True)
            
        if cache_type == "embeddings" or cache_type is None:
            removed += sum(1 for _ in Path(self.embeddings_dir).glob("*.pkl") if _.unlink() or True)
            
        if cache_type == "queries" or cache_type is None:
            removed += sum(1 for _ in Path(self.queries_dir).glob("*.pkl") if _.unlink() or True)
            
        logger.info(f"Removidos {removed} arquivos de cache" + 
                  (f" do tipo {cache_type}" if cache_type else ""))
        
        return removed


# Decoradores para uso com funções
def cached_description(cache_manager: CacheManager):
    """
    Decorador para cachear descrições de imagens.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, image_data, is_path=True, *args, **kwargs):
            if is_path:
                cached_result = cache_manager.get_cached_description(image_data)
                if cached_result:
                    return cached_result
                
                result = func(self, image_data, is_path, *args, **kwargs)
                cache_manager.cache_description(image_data, result)
                return result
            else:
                # Não podemos cachear bytes diretamente
                return func(self, image_data, is_path, *args, **kwargs)
        return wrapper
    return decorator

def cached_embedding(cache_manager: CacheManager):
    """
    Decorador para cachear embeddings de textos.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, text, *args, **kwargs):
            cached_result = cache_manager.get_cached_embedding(text)
            if cached_result:
                return cached_result
            
            result = func(self, text, *args, **kwargs)
            cache_manager.cache_embedding(text, result)
            return result
        return wrapper
    return decorator

def cached_query(cache_manager: CacheManager):
    """
    Decorador para cachear resultados de consultas.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, query_text, limit=5, *args, **kwargs):
            cached_result = cache_manager.get_cached_query_results(query_text, limit)
            if cached_result:
                return cached_result
            
            result = func(self, query_text, limit, *args, **kwargs)
            cache_manager.cache_query_results(query_text, limit, result)
            return result
        return wrapper
    return decorator