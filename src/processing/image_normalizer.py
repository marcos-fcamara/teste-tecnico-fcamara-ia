import os
import logging
from typing import Tuple, Optional, List, Dict
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)

class ImageNormalizer:
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 enhance_contrast: bool = True,
                 sharpen: bool = True,
                 normalize_lighting: bool = True):
        
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
        self.sharpen = sharpen
        self.normalize_lighting = normalize_lighting
        
    def normalize_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Normaliza uma imagem para melhorar a qualidade para processamento de visão computacional.
        
        Args:
            image_path: Caminho para a imagem original
            output_path: Caminho para salvar a imagem normalizada (opcional)
            
        Returns:
            Caminho para a imagem normalizada
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
            
            if not output_path:
                dir_path = os.path.dirname(image_path)
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(dir_path, f"{name}_normalized{ext}")
            
            if os.path.exists(output_path):
                return output_path
            
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                width, height = img.size
                if width < 200 or height < 200:
                    logger.info(f"Realizando upscaling da imagem {image_path} ({width}x{height})")
                    img = self._upscale_image(img)
                
                img = ImageOps.contain(img, self.target_size)
                
                if self.enhance_contrast:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)
                    
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.1)
                
                if self.sharpen:
                    img = img.filter(ImageFilter.SHARPEN)
                
                if self.normalize_lighting:
                    img = self._normalize_lighting(img)
                
                img.save(output_path, quality=95)
                logger.info(f"Imagem normalizada salva em: {output_path}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Erro ao normalizar imagem {image_path}: {str(e)}")
            return image_path
    
    def normalize_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Normaliza todas as imagens em um diretório.
        
        Args:
            input_dir: Diretório de entrada com imagens originais
            output_dir: Diretório de saída para imagens normalizadas (opcional)
            
        Returns:
            Dicionário mapeando caminhos originais para normalizados
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        path_mapping = {}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_extensions:
                if output_dir:
                    out_path = os.path.join(output_dir, filename)
                else:
                    out_path = None
                
                normalized_path = self.normalize_image(file_path, out_path)
                path_mapping[file_path] = normalized_path
        
        logger.info(f"Normalizadas {len(path_mapping)} imagens de {input_dir}")
        return path_mapping
    
    def _upscale_image(self, img: Image.Image) -> Image.Image:
        """Aplica upscaling em imagens pequenas para melhorar a qualidade."""
        width, height = img.size
        scale_factor = min(2.5, max(512 / width, 512 / height))
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def _normalize_lighting(self, img: Image.Image) -> Image.Image:
        """Normaliza a iluminação da imagem usando equalização de histograma."""
        img_array = np.array(img)
        
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        
        y_eq = self._equalize_hist(y)
        
        alpha = 0.7
        y_final = alpha * y_eq + (1 - alpha) * y
        
        factor_r = y_final / (y + 1e-10)
        factor_g = y_final / (y + 1e-10)
        factor_b = y_final / (y + 1e-10)
        
        r_new = np.clip(r * factor_r, 0, 255).astype(np.uint8)
        g_new = np.clip(g * factor_g, 0, 255).astype(np.uint8)
        b_new = np.clip(b * factor_b, 0, 255).astype(np.uint8)
        
        result_array = np.stack((r_new, g_new, b_new), axis=2)
        
        return Image.fromarray(result_array)
    
    def _equalize_hist(self, img_array: np.ndarray) -> np.ndarray:
        """Equalização de histograma personalizada."""
        flat = img_array.flatten()       
        hist, bins = np.histogram(flat, 256, [0, 256])  
        cdf = hist.cumsum()   
        cdf_normalized = cdf * 255 / cdf[-1]    
        equalized = np.interp(flat, bins[:-1], cdf_normalized)    
        return equalized.reshape(img_array.shape)