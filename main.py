import os
import sys
import logging
import argparse
import uvicorn
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import API_HOST, API_PORT
from src.processing.indexer import ImageIndexer
from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def index_images(args):
    """Indexar imagens no banco vetorial."""
    logger.info(f"Iniciando indexação de imagens em {args.images_dir}")
    
    vector_db = VectorDatabase()
    image_processor = ImageProcessor()
    
    indexer = ImageIndexer(
        image_processor=image_processor,
        vector_db=vector_db,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    success = indexer.index_images(args.images_dir, rebuild_index=args.rebuild)
    
    if success:
        logger.info("Indexação concluída com sucesso")
        db_info = vector_db.get_collection_info()
        logger.info(f"Informações do banco: {db_info}")
    else:
        logger.error("Falha na indexação")
        sys.exit(1)

def search_images(args):
    """Realizar busca por texto no banco vetorial."""
    
    logger.info(f"Realizando busca por: '{args.query}'")
    
    vector_db = VectorDatabase()
    image_processor = ImageProcessor()
    
    indexer = ImageIndexer(
        image_processor=image_processor,
        vector_db=vector_db
    )
    
    results = indexer.search_by_text(args.query, args.limit)
    
    if "error" in results:
        logger.error(f"Erro na busca: {results['error']}")
        sys.exit(1)
    
    raw_results = results["results"]
    
    print(f"\nResultados para a busca: '{args.query}'")
    print("-" * 50)
    
    for i in range(len(raw_results["ids"][0])):
        image_id = raw_results["ids"][0][i]
        distance = raw_results["distances"][0][i]
        similarity = 1 - distance
        
        metadata = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
        
        print(f"Imagem: {image_id}")
        print(f"Similaridade: {similarity:.4f}")
        print(f"Caminho: {metadata.get('path', 'N/A')}")
        print("-" * 50)

def run_api(args):
    """Iniciar a API FastAPI."""
    from src.api.app import app
    
    logger.info(f"Iniciando API em {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(description="Sistema de Busca por Similaridade de Imagens")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    index_parser = subparsers.add_parser("index", help="Indexar imagens no banco vetorial")
    index_parser.add_argument("--images-dir", type=str, default="src/data/images", 
                             help="Diretório contendo as imagens a serem indexadas")
    index_parser.add_argument("--rebuild", action="store_true", 
                             help="Reconstruir o índice do zero")
    index_parser.add_argument("--batch-size", type=int, default=10, 
                             help="Tamanho do lote para processamento")
    index_parser.add_argument("--workers", type=int, default=4, 
                             help="Número máximo de workers para processamento paralelo")
    
    search_parser = subparsers.add_parser("search", help="Buscar imagens por texto")
    search_parser.add_argument("query", type=str, help="Texto de consulta")
    search_parser.add_argument("--limit", type=int, default=5, 
                              help="Número máximo de resultados")
    
    api_parser = subparsers.add_parser("api", help="Iniciar a API")
    api_parser.add_argument("--host", type=str, default=API_HOST, 
                           help="Host para a API")
    api_parser.add_argument("--port", type=int, default=API_PORT, 
                           help="Porta para a API")
    api_parser.add_argument("--reload", action="store_true", 
                           help="Ativar reload automático na API")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("API key da OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY.")
        sys.exit(1)
    
    if args.command == "index":
        index_images(args)
    elif args.command == "search":
        search_images(args)
    elif args.command == "api":
        run_api(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()