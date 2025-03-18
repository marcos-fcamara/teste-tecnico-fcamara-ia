import os
import base64
import logging
from typing import Dict, List, Any, Optional, Union
import time
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI

from src.utils.cache_manager import CacheManager, cached_description, cached_embedding

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Classe para processar imagens e extrair embeddings e descrições."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o processador de imagens.
        
        Args:
            api_key: Chave da API da OpenAI. Se None, será usada a variável de ambiente.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key da OpenAI não fornecida")
            
        self.client = OpenAI(api_key=self.api_key)
        self.vision_model = os.getenv("VISION_MODEL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        
        # Inicializa o gerenciador de cache
        self.cache_manager = CacheManager()
    
    def _encode_image(self, image_path: str) -> str:
        """
        Codifica a imagem em base64.
        
        Args:
            image_path: Caminho para a imagem.
            
        Returns:
            str: Imagem codificada em base64.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """
        Codifica bytes de imagem em base64.
        
        Args:
            image_bytes: Bytes da imagem.
            
        Returns:
            str: Imagem codificada em base64.
        """
        return base64.b64encode(image_bytes).decode('utf-8')
    
    @cached_description(CacheManager())
    def generate_image_description(self, 
                                  image_data: Union[str, bytes], 
                                  is_path: bool = True) -> str:
        """
        Gera uma descrição detalhada da imagem usando OpenAI Vision.
        
        Args:
            image_data: Caminho da imagem ou bytes da imagem.
            is_path: Se True, image_data é um caminho. Se False, são bytes.
            
        Returns:
            str: Descrição detalhada da imagem.
        """
        try:
            # Verificar cache primeiro
            if is_path:
                cached_description = self.cache_manager.get_cached_description(image_data)
                if cached_description:
                    logger.info(f"Usando descrição em cache para {image_data}")
                    return cached_description
                    
            # Codifica a imagem conforme o tipo de entrada
            if is_path:
                base64_image = self._encode_image(image_data)
            else:
                base64_image = self._encode_image_from_bytes(image_data)
            
            # Extrai o nome do arquivo se estiver disponível
            filename = ""
            if is_path:
                filename = os.path.basename(image_data)
                # Limpa o nome do arquivo para extrair informações úteis
                filename = filename.replace('_220x220', '').replace('.jpg', '').replace('-', ' ').replace('_', ' ')
            
            logger.info(f"Gerando descrição para imagem: {os.path.basename(image_data) if is_path else 'bytes'}")
            
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um especialista em moda com foco em análise detalhada de vestuário. "
                            "Sua tarefa é analisar imagens de roupas e fornecer descrições estruturadas e precisas. "
                            "Seja extremamente detalhado nas descrições de cores, padrões, estilos e características."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"""Descreva detalhadamente esta peça de roupa. 
Nome do arquivo: {filename}

Inclua especificamente as seguintes informações em formato estruturado:
- Tipo de peça (exemplo: camisa, calça, vestido)
- Ocasião de uso (exemplo: casual, formal, esportiva)
- Cores predominantes
- Padrão (exemplo: liso, listrado, floral)
- Materiais aparentes
- Estilo/estética
- Gênero (masculino, feminino, unissex)
- Estação do ano mais adequada

Formate a resposta como um objeto JSON com estes campos e inclua um campo 'descrição_completa' com uma descrição narrativa.
Seja extremamente detalhado e específico."""},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            description = response.choices[0].message.content
            logger.debug(f"Descrição gerada com sucesso: {description[:50]}...")
            
            # Salva no cache se for um caminho de arquivo
            if is_path:
                self.cache_manager.cache_description(image_data, description)
                
            return description
            
        except Exception as e:
            logger.error(f"Erro ao gerar descrição da imagem: {str(e)}")
            return "Erro ao processar a imagem."
    
    @cached_embedding(CacheManager())
    def generate_embedding_from_text(self, text: str) -> List[float]:
        """
        Gera embeddings a partir de um texto usando a API da OpenAI.
        
        Args:
            text: Texto para gerar o embedding.
            
        Returns:
            List[float]: Vetor de embedding.
        """
        try:
            # Verificar cache primeiro
            cached_embedding = self.cache_manager.get_cached_embedding(text)
            if cached_embedding:
                logger.info(f"Usando embedding em cache")
                return cached_embedding
                
            logger.info("Gerando embedding para texto")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Salva no cache
            self.cache_manager.cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding do texto: {str(e)}")
            raise
    
    def process_image(self, 
                     image_path: str, 
                     image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa uma imagem, gerando descrição e embedding.
        
        Args:
            image_path: Caminho para a imagem.
            image_id: ID único para a imagem. Se None, usa o nome do arquivo.
            
        Returns:
            Dict: Dados processados da imagem.
        """
        if not image_id:
            image_id = os.path.basename(image_path)
            
        try:
            # Gera a descrição da imagem
            description = self.generate_image_description(image_path)
            
            # Gera o embedding a partir da descrição
            embedding = self.generate_embedding_from_text(description)
            
            # Extrai metadados básicos da imagem
            with Image.open(image_path) as img:
                width, height = img.size
                format_ = img.format
                mode = img.mode
            
            return {
                "id": image_id,
                "path": image_path,
                "description": description,
                "embedding": embedding,
                "metadata": {
                    "filename": os.path.basename(image_path),
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "format": format_,
                    "mode": mode
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            raise