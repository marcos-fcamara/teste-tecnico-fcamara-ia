import os
import logging
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase

logger = logging.getLogger(__name__)

class ImageIndexer:
    def __init__(self, 
                 image_processor: Optional[ImageProcessor] = None,
                 vector_db: Optional[VectorDatabase] = None,
                 batch_size: int = 10,
                 max_workers: int = 4):
        
        self.image_processor = image_processor or ImageProcessor()
        self.vector_db = vector_db or VectorDatabase()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")
        
        # Criar pasta de cache se não existir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Processa uma única imagem para obter descrição e embedding."""
        filename = os.path.basename(image_path)
        cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(filename)[0]}_cache.json")
        
        # Verificar se já existe no cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao ler cache para {filename}: {str(e)}. Processando novamente.")
        
        # Processar a imagem
        try:
            # Obter descrição
            description = self.image_processor.get_image_description(image_path)
            
            # Criar texto consolidado para embedding
            if isinstance(description, dict) and "error" not in description:
                # Criar um texto enriquecido com formato estruturado
                description_parts = []
                for key, value in description.items():
                    if value and isinstance(value, str):
                        description_parts.append(f"{key}: {value}")
                    elif value:
                        description_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                
                # Adicionar frases com combinações de atributos para melhorar a semântica
                if "Tipo de peça" in description and "Cores predominantes" in description:
                    tipo = description["Tipo de peça"]
                    cores = description["Cores predominantes"]
                    description_parts.append(f"{tipo} {cores}")
                
                if "Tipo de peça" in description and "Estilo/estética" in description:
                    tipo = description["Tipo de peça"]
                    estilo = description["Estilo/estética"]
                    description_parts.append(f"{tipo} com estilo {estilo}")
                
                if "Ocasião de uso" in description and "Estação do ano mais adequada" in description:
                    ocasiao = description["Ocasião de uso"]
                    estacao = description["Estação do ano mais adequada"]
                    description_parts.append(f"Para {ocasiao} na {estacao}")
                
                description_text = " ".join(description_parts)
            else:
                description_text = str(description)
            
            # Obter embedding
            embedding = self.image_processor.get_text_embedding(description_text)
            
            # Criar resultado
            result = {
                "id": filename,
                "path": image_path,
                "description": description,
                "description_text": description_text,
                "embedding": embedding
            }
            
            # Salvar em cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Erro ao salvar cache para {filename}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento de {filename}: {str(e)}")
            return {"id": filename, "path": image_path, "error": str(e)}
    
    def process_images_parallel(self, image_folder: str) -> List[Dict[str, Any]]:
        """Processa imagens em paralelo usando ThreadPoolExecutor."""
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_folder}")
        
        # Listar arquivos de imagem
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = [
            os.path.join(image_folder, filename)
            for filename in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, filename)) and 
            os.path.splitext(filename)[1].lower() in valid_extensions
        ]
        
        total_images = len(image_paths)
        logger.info(f"Iniciando processamento de {total_images} imagens em {image_folder}")
        
        results = []
        
        # Processar em paralelo com barra de progresso
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter tarefas
            future_to_path = {executor.submit(self._process_image, path): path for path in image_paths}
            
            # Processar resultados à medida que são concluídos
            for future in tqdm(as_completed(future_to_path), total=total_images, desc="Processando imagens"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Exceção não tratada para {path}: {str(e)}")
        
        logger.info(f"Processamento concluído: {len(results)}/{total_images} imagens processadas com sucesso")
        return results
    
    def index_images(self, image_folder: str, rebuild_index: bool = False) -> bool:
        """Processa e indexa todas as imagens em uma pasta no banco de vetores."""
        try:
            # Verificar se deve reconstruir o índice
            if rebuild_index:
                logger.info("Reconstruindo índice - apagando coleção existente")
                self.vector_db.delete_collection()
                # Reconectar ao banco
                self.vector_db = VectorDatabase(
                    persist_directory=self.vector_db.persist_directory,
                    collection_name=self.vector_db.collection_name
                )
            
            # Processar imagens
            processed_images = self.process_images_parallel(image_folder)
            
            # Filtrar imagens com erro
            valid_images = [img for img in processed_images if "error" not in img]
            error_count = len(processed_images) - len(valid_images)
            
            if error_count > 0:
                logger.warning(f"{error_count} imagens não puderam ser processadas")
            
            # Adicionar em lotes ao banco vetorial
            total_batches = (len(valid_images) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(valid_images), self.batch_size):
                batch = valid_images[i:i+self.batch_size]
                
                # Preparar dados para inserção
                ids = [item["id"] for item in batch]
                embeddings = [item["embedding"] for item in batch]
                documents = [item.get("description_text", "") for item in batch]
                
                # Preparar metadados
                metadatas = []
                for item in batch:
                    metadata = {
                        "path": item["path"],
                        "filename": item["id"]
                    }
                    
                    # Adicionar campos da descrição ao metadata
                    if isinstance(item.get("description"), dict):
                        for key, value in item["description"].items():
                            # Garantir que os valores sejam strings
                            metadata[key] = str(value) if value is not None else ""
                    
                    metadatas.append(metadata)
                
                # Adicionar lote ao banco
                success = self.vector_db.add_items(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                if not success:
                    logger.error(f"Falha ao adicionar lote {i//self.batch_size + 1}/{total_batches}")
                else:
                    logger.info(f"Lote {i//self.batch_size + 1}/{total_batches} adicionado com sucesso")
            
            # Obter informações da coleção
            collection_info = self.vector_db.get_collection_info()
            logger.info(f"Indexação concluída: {collection_info['item_count']} itens na coleção")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro durante indexação: {str(e)}")
            return False
    
    def search_by_text(self, query_text: str, limit: int = 5) -> Dict[str, Any]:
        """Realiza uma busca por texto no banco vetorial."""
        try:
            # Converter texto para embedding
            query_embedding = self.image_processor.get_text_embedding(query_text)
            
            # Realizar busca
            results = self.vector_db.query(
                query_embedding=query_embedding,
                limit=limit
            )
            
            return {
                "query": query_text,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Erro na busca por texto: {str(e)}")
            return {
                "query": query_text,
                "error": str(e),
                "results": None
            }