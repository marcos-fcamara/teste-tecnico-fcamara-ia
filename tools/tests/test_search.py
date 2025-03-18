import os
import sys
import logging
import json
from tabulate import tabulate
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
output_dir = "../results/search"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

os.environ["CHROMA_PERSIST_DIRECTORY"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chroma_db'))

from src.processing.indexer import ImageIndexer
from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "roupa feminina de verão com estampa geométrica azul e branca",
    "conjunto cropped e shorts estilo boêmio",
    "look casual feminino azul e branco para verão",
    "macaquinho com estampa de oncinha para o verão",
    "peça feminina sensual com animal print",
    "roupa casual de verão com padrão de oncinha em tons marrons",
    "macaquinho branco liso feminino para verão",
    "roupa moderna branca casual feminina",
    "peça de algodão branca para o verão estilo moderno",
    "conjunto feminino estampado para clima quente",
    "macaquinho leve para dia quente",
    "roupa com estampa animal para mulher"
    "vestido vermelho elegante para ocasião formal",
]

def format_results_for_table(results, query):
    """Formata os resultados para exibição em tabela."""
    table_data = []
    
    if not results or "error" in results:
        return [["Erro", "Erro na consulta", "-", "-"]]
    
    raw_results = results["results"]
    
    for i in range(len(raw_results["ids"][0])):
        # Obtemos o ID para referência
        image_id = raw_results["ids"][0][i]
        
        # Obtemos os metadados
        metadata = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
        
        # Extraímos o caminho da imagem dos metadados
        image_path = metadata.get("path", image_id)
        
        # Para melhor visualização, mostramos o nome do arquivo
        image_filename = os.path.basename(image_path) if image_path else image_id
        
        distance = raw_results["distances"][0][i]
        similarity = 1 - distance
        
        document = raw_results["documents"][0][i] if raw_results["documents"] else ""
        
        # Extrair tipo de peça e cores da descrição em JSON
        tipo_peca = "N/A"
        cores = "N/A"
        
        try:
            # Verificar se o documento contém JSON
            import json
            import re
            
            # Tentar extrair o JSON da descrição
            json_match = re.search(r'\{.*\}', document, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(0))
                    
                    # Extrair tipo de peça (verificando várias possíveis chaves)
                    tipo_peca_keys = ["tipo_de_peca", "tipo_peca", "tipo"]
                    for key in tipo_peca_keys:
                        if key in json_data and json_data[key]:
                            tipo_peca = json_data[key]
                            break
                    
                    # Extrair cores (verificando várias possíveis chaves)
                    cores_keys = ["cores_predominantes", "cores", "cor"]
                    for key in cores_keys:
                        if key in json_data and json_data[key]:
                            if isinstance(json_data[key], list):
                                cores = ", ".join(json_data[key])
                            else:
                                cores = json_data[key]
                            break
                except:
                    # Se falhar ao processar o JSON, mantenha os valores padrão
                    pass
        except Exception as e:
            logger.debug(f"Erro ao processar metadados da imagem: {str(e)}")
        
        descricao = document[:100] + "..." if len(document) > 100 else document
        
        table_data.append([
            query,
            image_filename,
            f"{similarity:.4f}",
            tipo_peca,
            cores,
            descricao
        ])
    
    return table_data

def run_test_searches():
    """Executa buscas de teste e mostra os resultados em uma tabela."""
    try:
        vector_db = VectorDatabase()
        image_processor = ImageProcessor()
        
        indexer = ImageIndexer(
            image_processor=image_processor,
            vector_db=vector_db
        )
        
        db_info = vector_db.get_collection_info()
        if db_info["item_count"] == 0:
            logger.error("O banco de dados está vazio. Execute a indexação primeiro.")
            sys.exit(1)
        
        logger.info(f"Banco de dados contém {db_info['item_count']} itens indexados.")
        
        all_results = []
        
        for query in TEST_QUERIES:
            logger.info(f"Testando consulta: '{query}'")
            
            results = indexer.search_by_text(query, limit=3)
            
            table_data = format_results_for_table(results, query)
            all_results.extend(table_data)
        
        headers = ["Consulta", "Imagem", "Similaridade", "Tipo de Peça", "Cores", "Descrição"]
        print("\nResultados das Consultas de Teste")
        print(tabulate(all_results, headers=headers, tablefmt="grid"))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"test_results_{timestamp}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Resultados das Consultas de Teste\n\n")
            f.write(tabulate(all_results, headers=headers, tablefmt="grid"))
        
        logger.info(f"Resultados salvos em {output_file}")
        
    except Exception as e:
        logger.error(f"Erro durante os testes: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("API key da OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY.")
        sys.exit(1)
    
    run_test_searches()