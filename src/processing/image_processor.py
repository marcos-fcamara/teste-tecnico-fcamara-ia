import os
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI
import numpy as np
from PIL import Image
import io

from src.processing.image_normalizer import ImageNormalizer

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None,
                 normalize_images: bool = True):
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key da OpenAI não fornecida")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("VISION_MODEL")
        
        self.normalize_images = normalize_images
        if normalize_images:
            self.normalizer = ImageNormalizer(
                target_size=(512, 512),
                enhance_contrast=True,
                sharpen=True,
                normalize_lighting=True
            )
        
    def encode_image(self, image_path: str) -> str:
        """Codifica uma imagem em base64 para envio à API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Erro ao codificar imagem {image_path}: {str(e)}")
            raise
    
    def get_image_description(self, image_path: str) -> Dict[str, str]:
        """Obtém descrição detalhada da imagem usando a API de visão da OpenAI."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
            
        try:
            if self.normalize_images:
                try:
                    normalized_path = self.normalizer.normalize_image(image_path)
                    
                    processing_path = normalized_path
                    logger.info(f"Usando imagem normalizada: {normalized_path}")
                except Exception as e:
                    logger.warning(f"Falha ao normalizar imagem {image_path}: {str(e)}")
                    processing_path = image_path
            else:
                processing_path = image_path
            
            with Image.open(processing_path) as img:
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(x * ratio) for x in img.size)
                    img = img.resize(new_size, Image.LANCZOS)
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                buffer = io.BytesIO()
                img = img.convert('RGB')
                img.save(buffer, format="JPEG", quality=90)
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
            prompt = """
            Descreva detalhadamente esta peça de roupa. 
            Inclua especificamente as seguintes informações em formato estruturado:
            - Tipo de peça (exemplo: camisa, calça, vestido)
            - Ocasião de uso (exemplo: casual, formal, esportiva)
            - Cores predominantes
            - Padrão (exemplo: liso, listrado, floral)
            - Materiais aparentes
            - Estilo/estética
            - Gênero (masculino, feminino, unissex)
            - Estação do ano mais adequada
            
            Formate a resposta como um objeto JSON com estes campos.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em moda que descreve imagens de roupas com precisão e detalhes."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=500
            )
            
            description_text = response.choices[0].message.content
            
            try:
                import json
                import re
                
                json_match = re.search(r'\{.*\}', description_text, re.DOTALL)
                if json_match:
                    description_json = json.loads(json_match.group(0))
                else:
                    description_json = {"description": description_text}
                
                logger.info(f"Descrição obtida com sucesso para {os.path.basename(image_path)}")
                return description_json
                
            except json.JSONDecodeError:
                logger.warning(f"Resposta não está em formato JSON para {image_path}. Retornando texto bruto.")
                return {"description": description_text}
                
        except Exception as e:
            logger.error(f"Erro ao obter descrição para {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Obtém embedding de um texto usando a API de embeddings da OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Erro ao obter embedding de texto: {str(e)}")
            raise
    
    def process_image_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Processa todas as imagens em uma pasta e retorna uma lista de dicionários com metadados."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")
            
        image_data = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_extensions:
                try:
                    logger.info(f"Processando imagem: {filename}")
                    
                    description = self.get_image_description(file_path)
                    
                    if isinstance(description, dict) and "error" not in description:
                        description_text = " ".join(str(v) for v in description.values() if v)
                    else:
                        description_text = str(description)
                    
                    embedding = self.get_text_embedding(description_text)
                    
                    image_data.append({
                        "id": filename,
                        "path": file_path,
                        "description": description,
                        "embedding": embedding
                    })
                    
                except Exception as e:
                    logger.error(f"Erro ao processar {filename}: {str(e)}")
                    continue
        
        logger.info(f"Processadas {len(image_data)} imagens de {folder_path}")
        return image_data