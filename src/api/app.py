import os
import logging
import json
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.processing.indexer import ImageIndexer
from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sistema de Busca por Similaridade de Imagens",
    description="API para busca de imagens por similaridade através de consultas textuais",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class SearchResult(BaseModel):
    query: str
    results: List[Dict[str, Any]]

vector_db = None
image_processor = None
image_indexer = None
images_folder = os.getenv("IMAGES_FOLDER", "src/data/images")

if not os.path.exists(images_folder):
    logger.warning(f"Pasta de imagens não encontrada: {images_folder}")
    os.makedirs(images_folder, exist_ok=True)

def initialize_components():
    global vector_db, image_processor, image_indexer
    
    if vector_db is None:
        logger.info("Inicializando banco vetorial...")
        vector_db = VectorDatabase()
    
    if image_processor is None:
        logger.info("Inicializando processador de imagens...")
        image_processor = ImageProcessor()
    
    if image_indexer is None:
        logger.info("Inicializando indexador...")
        image_indexer = ImageIndexer(
            image_processor=image_processor,
            vector_db=vector_db,
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_workers=int(os.getenv("MAX_WORKERS", "4"))
        )
    
    return {
        "vector_db": vector_db,
        "image_processor": image_processor,
        "image_indexer": image_indexer
    }

def get_image_indexer():
    components = initialize_components()
    return components["image_indexer"]

def get_vector_db():
    components = initialize_components()
    return components["vector_db"]

def get_image_processor():
    components = initialize_components()
    return components["image_processor"]

static_path = Path("src/data/images")
if static_path.exists():
    app.mount("/images", StaticFiles(directory=str(static_path)), name="images")

@app.get("/")
def read_root():
    return {"message": "API de Busca por Similaridade de Imagens", "status": "online"}

@app.get("/health")
def health_check():
    try:
        components = initialize_components()
        db_info = components["vector_db"].get_collection_info()
        return {
            "status": "healthy",
            "database": db_info
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/index")
def index_images(
    rebuild: bool = Query(False, description="Reconstruir o índice do zero"),
    indexer: ImageIndexer = Depends(get_image_indexer)
):
    try:
        success = indexer.index_images(images_folder, rebuild_index=rebuild)
        if success:
            db_info = vector_db.get_collection_info()
            return {
                "status": "success",
                "message": "Indexação concluída com sucesso",
                "database": db_info
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Falha na indexação"
                }
            )
    except Exception as e:
        logger.error(f"Erro durante indexação: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/search", response_model=SearchResult)
def search_images(
    search_query: SearchQuery,
    indexer: ImageIndexer = Depends(get_image_indexer)
):
    try:
        search_results = indexer.search_by_text(search_query.query, search_query.limit)
        
        if "error" in search_results:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": search_results["error"]
                }
            )
        
        results = []
        raw_results = search_results["results"]
        
        if raw_results:
            for i in range(len(raw_results["ids"][0])):
                image_id = raw_results["ids"][0][i]
                distance = raw_results["distances"][0][i]
                
                similarity = 1 - distance
                
                metadata = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
                document = raw_results["documents"][0][i] if raw_results["documents"] else ""
                
                image_path = metadata.get("path", "")
                image_filename = metadata.get("filename", image_id)
                image_url = f"/images/{image_filename}"
                
                results.append({
                    "id": image_id,
                    "similarity": similarity,
                    "distance": distance,
                    "metadata": metadata,
                    "description": document,
                    "image_url": image_url
                })
        
        return {
            "query": search_query.query,
            "results": results
        }
            
    except Exception as e:
        logger.error(f"Erro na busca: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe")
def describe_image(
    image_path: str = Query(..., description="Caminho para a imagem a ser descrita"),
    processor: ImageProcessor = Depends(get_image_processor)
):
    try:
        if not os.path.isabs(image_path):
            full_path = os.path.join(images_folder, image_path)
        else:
            full_path = image_path
        
        if not os.path.exists(full_path):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"Imagem não encontrada: {full_path}"}
            )
        
        description = processor.get_image_description(full_path)
        
        return {
            "status": "success",
            "image_path": image_path,
            "description": description
        }
        
    except Exception as e:
        logger.error(f"Erro ao descrever imagem: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)