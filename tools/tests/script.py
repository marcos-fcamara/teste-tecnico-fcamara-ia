import os
import sys
import logging
import json
import chromadb

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_chroma_db():
    """Inspeciona o banco de dados ChromaDB diretamente."""
    # Caminho para o banco de dados ChromaDB
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chroma_db'))
    
    logger.info(f"Conectando ao banco de dados em: {db_path}")
    
    # Conectar diretamente ao banco de dados
    client = chromadb.PersistentClient(path=db_path)
    
    # Listar coleções
    collection_names = client.list_collections()
    
    if not collection_names:
        logger.error("Não há coleções no banco de dados.")
        return
    
    logger.info(f"Coleções disponíveis: {collection_names}")
    
    # Para cada coleção, examinar os primeiros itens
    for collection_name in collection_names:
        logger.info(f"\nInspecionando coleção: {collection_name}")
        
        collection = client.get_collection(collection_name)
        
        # Obter todos os itens (ou os primeiros 100 se for uma coleção grande)
        results = collection.peek(limit=100)
        
        total_items = len(results["ids"])
        logger.info(f"Total de itens na amostra: {total_items}")
        
        if total_items == 0:
            logger.info("A coleção está vazia ou não retornou dados.")
            continue
        
        # Analisar estrutura da coleção
        logger.info(f"Estrutura do resultado: {list(results.keys())}")
        
        # Mostrar os primeiros 3 itens para análise
        logger.info("Primeiros 3 itens:")
        
        for i in range(min(3, total_items)):
            item_id = results["ids"][i]
            metadata = results.get("metadatas", [{}])[i] if results.get("metadatas") else {}
            document = results.get("documents", [""])[i] if results.get("documents") else ""
            
            logger.info(f"\nItem {i+1}:")
            logger.info(f"ID: {item_id}")
            
            # Mostrar metadados (limitados a 300 caracteres para legibilidade)
            metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
            logger.info(f"Metadata: {metadata_str[:300]}..." if len(metadata_str) > 300 else f"Metadata: {metadata_str}")
            
            # Mostrar início do documento
            logger.info(f"Document (primeiros 300 caracteres): {document[:300]}...")
            
            # Buscar referências a caminhos de imagem em várias formas
            for field in ["image_path", "imagePath", "path", "filepath", "file"]:
                if field in metadata:
                    logger.info(f"Caminho de imagem encontrado em metadata['{field}']: {metadata[field]}")
            
            # Buscar no documento usando expressões comuns
            for pattern in ["image_path", "src/data/images", ".jpg", ".png"]:
                if pattern in document:
                    pos = document.find(pattern)
                    snippet = document[max(0, pos-20):min(len(document), pos+100)]
                    logger.info(f"Trecho relevante com '{pattern}': '{snippet}'")

if __name__ == "__main__":
    inspect_chroma_db()