import os
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import json
import numpy as np

from src.database.vector_db import VectorDatabase
from src.processing.image_processor import ImageProcessor
from src.embeddings.text_processor import TextProcessor
from src.utils.cache_manager import CacheManager, cached_query

logger = logging.getLogger(__name__)


class ImageIndexer:
    """Classe para indexar imagens no banco vetorial."""

    def __init__(
        self,
        image_processor: ImageProcessor,
        vector_db: VectorDatabase,
        batch_size: int = 10,
        max_workers: int = 4,
    ):
        """
        Inicializa o indexador de imagens.

        Args:
            image_processor: Instância do processador de imagens.
            vector_db: Instância do banco de dados vetorial.
            batch_size: Tamanho do lote para processamento.
            max_workers: Número máximo de workers para processamento paralelo.
        """
        self.image_processor = image_processor
        self.vector_db = vector_db
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.text_processor = TextProcessor()
        self.cache_manager = CacheManager()

    def _process_image(self, image_path: str, image_id: str = None) -> Dict[str, Any]:
        """
        Processa uma imagem individual.

        Args:
            image_path: Caminho para a imagem.
            image_id: ID opcional para a imagem.

        Returns:
            Dict: Dados da imagem processada.
        """
        try:
            return self.image_processor.process_image(image_path, image_id)
        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            return None

    def enhance_query(self, query: str) -> str:
        """
        Método para aprimorar consultas de busca.
        
        Args:
            query (str): Consulta original

        Returns:
            str: Consulta aprimorada
        """

        stopwords = {'de', 'para', 'com', 'e', 'o', 'a', 'os', 'as'}
        palavras = query.lower().split()
        palavras_filtradas = [p for p in palavras if p not in stopwords]
        
        sinonimos = {
            'roupa': ['vestimenta', 'traje'],
            'feminina': ['feminino', 'mulher'],
            'verão': ['estival', 'sazonal'],
            'casual': ['informal', 'despojado']
        }
        
        aprimored_search = []
        for palavra in palavras_filtradas:
            aprimored_search.append(palavra)
            aprimored_search.extend(sinonimos.get(palavra, []))
        
        return ' '.join(aprimored_search)

    def _find_images(self, directory: str) -> List[str]:
        """
        Encontra arquivos de imagem em um diretório.

        Args:
            directory: Diretório para buscar imagens.

        Returns:
            List[str]: Lista de caminhos das imagens.
        """
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
        image_paths = []

        for ext in valid_extensions:
            pattern = f"*{ext}"
            image_paths.extend([str(p) for p in Path(directory).glob(pattern)])
            image_paths.extend([str(p) for p in Path(directory).glob(pattern.upper())])

        logger.info(f"Encontradas {len(image_paths)} imagens em {directory}")
        return image_paths

    def index_images(self, directory: str, rebuild_index: bool = False) -> bool:
        """
        Indexa todas as imagens em um diretório.

        Args:
            directory: Diretório contendo as imagens.
            rebuild_index: Se True, remove o índice existente antes de reindexar.

        Returns:
            bool: True se a operação foi bem-sucedida.
        """
        if not os.path.isdir(directory):
            logger.error(f"Diretório não encontrado: {directory}")
            return False

        if rebuild_index:
            logger.info("Removendo índice existente...")
            self.vector_db.delete_collection()

            logger.info("Limpando cache...")
            self.cache_manager.clear_cache()

            time.sleep(1) 

        image_paths = self._find_images(directory)

        if not image_paths:
            logger.warning(f"Nenhuma imagem encontrada em {directory}")
            return False

        processed_count = 0
        batch_count = 0

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_count += 1

            logger.info(
                f"Processando lote {batch_count}/{(len(image_paths) + self.batch_size - 1) // self.batch_size}..."
            )

            processed_batch = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_image, path, f"img_{i + idx}"): path
                    for idx, path in enumerate(batch_paths, 1)
                }

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            processed_batch.append(result)
                    except Exception as e:
                        logger.error(f"Falha ao processar {path}: {str(e)}")

            if processed_batch:
                ids = [item["id"] for item in processed_batch]
                embeddings = [item["embedding"] for item in processed_batch]
                documents = [item["description"] for item in processed_batch]
                metadatas = [item["metadata"] for item in processed_batch]

                success = self.vector_db.add_items(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                if success:
                    processed_count += len(processed_batch)
                    logger.info(
                        f"Lote {batch_count} adicionado com sucesso. Total processado: {processed_count}/{len(image_paths)}"
                    )
                else:
                    logger.error(
                        f"Falha ao adicionar lote {batch_count} ao banco vetorial."
                    )

            if (
                batch_count
                < (len(image_paths) + self.batch_size - 1) // self.batch_size
            ):
                time.sleep(1)

        cache_stats = self.cache_manager.get_cache_stats()
        logger.info(f"Estatísticas do cache: {cache_stats}")

        logger.info(
            f"Indexação concluída. {processed_count}/{len(image_paths)} imagens processadas com sucesso."
        )
        return processed_count > 0

    @cached_query(CacheManager())
    def search_by_text(
        self, query_text: str, limit: int = 5, high_quality: bool = True, reuse_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Busca imagens por texto com técnicas avançadas para alta similaridade.

        Args:
            query_text: Texto da consulta.
            limit: Número máximo de resultados.
            high_quality: Se True, usa técnicas avançadas para maximizar a similaridade.
            reuse_embeddings: Se True, usa apenas embeddings em cache sem chamar a API.

        Returns:
            Dict: Resultados da busca.
        """
        try:
            logger.info(f"Iniciando busca para consulta: '{query_text}'")
            logger.info(f"Modo de alta qualidade: {high_quality}")
            logger.info(f"Limite de resultados: {limit}")
            logger.info(f"Modo reuse_embeddings: {reuse_embeddings}")
            
            if reuse_embeddings:
                cached_query = self.cache_manager.get_cached_query_results(
                    f"enhance_v2_{query_text}", 1
                )
                if cached_query:
                    logger.info(f"Usando consulta aprimorada em cache")
                    enhanced_query = cached_query.get("enhanced_query", query_text)
                else:
                    logger.info(f"Consulta não encontrada no cache e modo reuse ativado. Usando consulta original.")
                    enhanced_query = query_text
            else:
                if callable(getattr(self.text_processor, 'enhance_query', None)):
                    enhanced_query = self.text_processor.enhance_query(query_text)
                else:
                    enhanced_query = self.enhance_query(query_text)
            
            cache_key = f"high_quality_{query_text}" if high_quality else query_text
            cached_results = self.cache_manager.get_cached_query_results(
                cache_key, limit
            )
            if cached_results:
                logger.info(f"Usando resultados em cache para consulta: '{query_text}'")
                return cached_results

            if high_quality:
                query_embedding = self.text_processor.generate_embedding(
                    enhanced_query, 
                    use_ensemble=True,
                    force_local=reuse_embeddings
                )
                
                if reuse_embeddings and query_embedding is None:
                    logger.warning(f"Modo reuse_embeddings ativado, mas embedding não encontrado no cache")
                    query_embedding = self.text_processor.generate_embedding(
                        enhanced_query, 
                        use_ensemble=True,
                        force_local=False
                    )

                    return {
                        "error": "Embedding não encontrado no cache e modo reuse_embeddings ativado"
                    }

                logger.info(f"Processando consulta avançada: '{query_text}'")


                feature_weights = {
                    "tipo_peca": 2.0,
                    "cor": 1.5,
                    "padrao": 1.2,
                    "estilo": 1.8, 
                    "ocasiao": 1.5,
                    "genero": 1.3,
                    "estacao": 1.0,
                    "material": 1.0,
                }

                if high_quality:
                    enhanced_query = self.text_processor.enhance_query(query_text)

                    query_embedding = self.text_processor.generate_embedding(
                        enhanced_query, use_ensemble=True
                    )

                    initial_results = self.vector_db.query(
                        query_embedding=query_embedding,
                        limit=limit,
                        feature_weights=feature_weights,
                    )

                    if len(initial_results["ids"][0]) > 0:
                        final_results = {
                            "ids": initial_results["ids"].copy(),
                            "documents": initial_results["documents"].copy(),
                            "metadatas": initial_results["metadatas"].copy(),
                            "distances": initial_results["distances"].copy(),
                        }

                        for i in range(len(initial_results["ids"][0])):
                            try:
                                doc = initial_results["documents"][0][i]

                                json_match = re.search(r"\{.*\}", doc, re.DOTALL)

                                if json_match:
                                    try:
                                        json_data = json.loads(json_match.group(0))
                                        relevant_parts = []

                                        for field in [
                                            "tipo_peca",
                                            "cores_predominantes",
                                            "padrao",
                                            "estilo",
                                            "ocasiao_uso",
                                            "genero",
                                            "estacao",
                                            "materiais",
                                            "descricao_completa",
                                        ]:
                                            if field in json_data and json_data[field]:
                                                if isinstance(json_data[field], list):
                                                    relevant_parts.append(
                                                        f"{field}: {', '.join(json_data[field])}"
                                                    )
                                                else:
                                                    relevant_parts.append(
                                                        f"{field}: {json_data[field]}"
                                                    )

                                        relevant_text = " ".join(relevant_parts)
                                    except:
                                        relevant_text = doc[:500]
                                else:
                                    relevant_text = doc[:500]

                                similarity_to_query = self._calculate_similarity(
                                    enhanced_query, relevant_text
                                )

                                original_distance = initial_results["distances"][0][i]

                                bidirectional_distance = (
                                    original_distance * 0.75
                                    + (1 - similarity_to_query) * 0.25
                                )

                                final_results["distances"][0][i] = bidirectional_distance

                            except Exception as e:
                                logger.debug(
                                    f"Erro ao processar similaridade bidirecional: {str(e)}"
                                )

                        items = [
                            (i, final_results["distances"][0][i])
                            for i in range(len(final_results["distances"][0]))
                        ]

                        items.sort(key=lambda x: x[1])

                        for key in ["ids", "documents", "metadatas", "distances"]:
                            final_results[key] = [
                                [final_results[key][0][item[0]] for item in items[:limit]]
                            ]

                        for i in range(len(final_results["distances"][0])):
                            distance = final_results["distances"][0][i]
                            similarity = 1 - distance

                            if similarity > 0.85:
                                amplified_similarity = (
                                    similarity + (0.98 - similarity) * 0.8
                                )
                                final_results["distances"][0][i] = 1 - amplified_similarity

                        result_data = {
                            "query": query_text,
                            "enhanced_query": enhanced_query,
                            "results": final_results,
                        }
                    else:
                        result_data = {
                            "query": query_text,
                            "enhanced_query": enhanced_query,
                            "results": initial_results,
                        }
                
            else:
                if reuse_embeddings:
                    query_embedding = self.text_processor.generate_embedding(
                        enhanced_query, 
                        use_ensemble=False,
                        force_local=True
                    )
                    
                    if query_embedding is None:
                        logger.warning(f"Modo reuse_embeddings ativado, mas embedding não encontrado no cache")
                        return {
                            "error": "Embedding não encontrado no cache e modo reuse_embeddings ativado"
                        }
                        
                    query_data = {
                        "embedding": query_embedding,
                        "enhanced_query": enhanced_query
                    }
                else:
                    query_data = self.text_processor.process_query(query_text, enhance=True)
                    
                results = self.vector_db.query(
                    query_embedding=query_data["embedding"], limit=limit
                )

                result_data = {
                    "query": query_text,
                    "enhanced_query": query_data["enhanced_query"],
                    "results": results,
                }

            self.cache_manager.cache_query_results(cache_key, limit, result_data)

            return result_data

        except Exception as e:
            logger.error(f"Erro na busca por texto: {str(e)}")
            return {"error": str(e)}

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula a similaridade entre dois textos usando embeddings.

        Args:
            text1: Primeiro texto.
            text2: Segundo texto.

        Returns:
            float: Valor de similaridade (0-1).
        """
        try:
            embedding1 = self.text_processor.generate_embedding(
                text1, use_ensemble=False
            )
            embedding2 = self.text_processor.generate_embedding(
                text2, use_ensemble=False
            )


            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 > 0 and norm2 > 0:
                embedding1 = [e / norm1 for e in embedding1]
                embedding2 = [e / norm2 for e in embedding2]

            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            return max(0.0, min(1.0, dot_product))

        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {str(e)}")
            return 0.0
