import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
import json
import re
import glob
import shutil

# Configurar logging detalhado
# Já configurado acima

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
os.environ["CHROMA_PERSIST_DIRECTORY"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chroma_db'))

from src.processing.indexer import ImageIndexer
from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase
from src.config import OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_queries(queries_file: str = "../data/input/test_queries.txt") -> List[str]:
    """Carrega consultas de teste de um arquivo."""
    if not os.path.exists(queries_file):
        logger.error(f"Arquivo de consultas não encontrado: {queries_file}")
        return []
    
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if '. ' in line:
                    line = line.split('. ', 1)[1]
                queries.append(line)
    
    return queries

def run_embedding_tests(queries: List[str], top_k: int = 3, output_dir: str = "../results/embeddings"):
    """Executa testes de embeddings para as consultas fornecidas."""
    if not queries:
        logger.error("Nenhuma consulta para testar.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    results_data = []
    
    for query in tqdm(queries, desc="Testando consultas"):
        logger.info(f"Testando consulta: '{query}'")
        
        search_results = indexer.search_by_text(query, limit=top_k)
        
        if "error" in search_results:
            logger.error(f"Erro na consulta '{query}': {search_results['error']}")
            continue
        
        raw_results = search_results["results"]
        
        # Log detalhado para verificar a estrutura
        logger.info(f"Estrutura dos resultados: {list(raw_results.keys())}")
        if "metadatas" in raw_results and raw_results["metadatas"] and len(raw_results["metadatas"]) > 0:
            logger.info(f"Exemplo de metadata: {raw_results['metadatas'][0][0] if raw_results['metadatas'][0] else 'Vazio'}")
        
        for i in range(len(raw_results["ids"][0])):
            image_id = raw_results["ids"][0][i]
            distance = raw_results["distances"][0][i]
            similarity = 1 - distance
            
            metadata = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
            document = raw_results["documents"][0][i] if raw_results["documents"] else ""
            
            # Extrair informações usando regex - abordagem mais robusta
            tipo_peca = "N/A"
            cores = "N/A"
            image_path = "N/A"
            
            # Obter caminho da imagem diretamente dos metadados
            if metadata and "path" in metadata:
                image_path = metadata["path"]
                # Extrair filename do path
                filename = os.path.basename(image_path)
            else:
                # Tentar extrair do documento como fallback
                image_path_match = re.search(r'"image_path"\s*:\s*"([^"]+)"', document)
                if image_path_match:
                    image_path = image_path_match.group(1)
                    filename = os.path.basename(image_path)
                else:
                    filename = f"imagem_{image_id}.jpg"
            
            # Extrair tipo_peca do documento
            tipo_peca_match = re.search(r'"tipo_de_peca"\s*:\s*"([^"]+)"', document) or re.search(r'"Tipo de peça"\s*:\s*"([^"]+)"', document)
            if tipo_peca_match:
                tipo_peca = tipo_peca_match.group(1)
            
            # Extrair cores do documento - pode ser lista ou string
            cores_match = re.search(r'"cores_predominantes"\s*:\s*(\[[^\]]+\])', document) or re.search(r'"Cores predominantes"\s*:\s*(\[[^\]]+\])', document)
            if cores_match:
                try:
                    cores_array = json.loads(cores_match.group(1))
                    if isinstance(cores_array, list):
                        cores = ", ".join(cores_array)
                except:
                    # Se falhar em transformar em lista, tentar extrair diretamente
                    cores_text = cores_match.group(1).strip('[]').replace('"', '')
                    cores = cores_text
            
            # Se ainda não tiver cores, procurar por versão string
            if cores == "N/A":
                cores_str_match = re.search(r'"cores_predominantes"\s*:\s*"([^"]+)"', document) or re.search(r'"Cores predominantes"\s*:\s*"([^"]+)"', document)
                if cores_str_match:
                    cores = cores_str_match.group(1)
            
            results_data.append({
                "query": query,
                "rank": i + 1,
                "image_id": image_id,
                "image_path": image_path,
                "filename": filename,
                "similarity": similarity,
                "tipo_peca": tipo_peca,
                "cores": cores,
                "description": document[:100] + "..." if len(document) > 100 else document
            })
    
    df = pd.DataFrame(results_data)
    
    output_csv = os.path.join(output_dir, "embedding_test_results.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Resultados salvos em {output_csv}")
    
    generate_test_report(df, output_dir)
    
    return df

def generate_test_report(results_df: pd.DataFrame, output_dir: str):
    """Gera relatório de teste a partir dos resultados."""
    if results_df.empty:
        logger.error("Sem dados para gerar relatório.")
        return
    
    num_queries = results_df['query'].nunique()
    avg_similarity = results_df['similarity'].mean()
    
    rank_similarity = results_df.groupby('rank')['similarity'].agg(['mean', 'min', 'max']).reset_index()
    
    report_html = os.path.join(output_dir, "embedding_test_report.html")
    
    # Criar diretório para imagens referenciadas no relatório
    images_report_dir = os.path.join(output_dir, "images")
    os.makedirs(images_report_dir, exist_ok=True)
    
    query_results = []
    for query in results_df['query'].unique():
        query_df = results_df[results_df['query'] == query].sort_values('rank')
        
        # Preparar a coluna de imagem para HTML
        html_images = []
        for idx, row in query_df.iterrows():
            original_path = row['image_path']
            filename = row['filename']
            
            logger.info(f"Processando imagem para HTML: path={original_path}, filename={filename}")
            
            # Verificar se o caminho existe 
            if os.path.exists(original_path):
                # Criar referência relativa para o HTML
                relative_path = f"images/{filename}"
                dest_path = os.path.join(images_report_dir, filename)
                
                # Copiar a imagem para o diretório do relatório
                try:
                    shutil.copy2(original_path, dest_path)
                    img_html = f'<img src="{relative_path}" alt="{row["tipo_peca"]}" width="150">'
                    logger.info(f"Imagem copiada com sucesso: {dest_path}")
                except Exception as e:
                    logger.error(f"Erro ao copiar imagem {original_path}: {str(e)}")
                    img_html = f'<img src="N/A" alt="Erro: {str(e)[:30]}..." width="150">'
            else:
                logger.warning(f"Caminho de imagem não existe: {original_path}")
                img_html = f'<img src="N/A" alt="Caminho não existe" width="150">'
            
            html_images.append(img_html)
            
            html_images.append(img_html)
        
        # Adicionar a coluna de HTML de imagens ao DataFrame
        temp_df = query_df.copy()
        temp_df['imagem'] = html_images
        
        # Gerar HTML da tabela
        query_table = temp_df[['rank', 'imagem', 'similarity', 'tipo_peca', 'cores']].to_html(
            index=False,
            escape=False,  # Importante: não escapar o HTML das imagens
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        
        query_results.append({
            "query": query,
            "avg_similarity": query_df['similarity'].mean(),
            "results_table": query_table
        })
    
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['similarity'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribuição de Similaridade')
    plt.xlabel('Similaridade')
    plt.ylabel('Frequência')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
    
    plt.figure(figsize=(10, 6))
    ranks = rank_similarity['rank'].values
    means = rank_similarity['mean'].values
    plt.bar(ranks, means, color='green', alpha=0.7)
    plt.title('Similaridade Média por Posição (Rank)')
    plt.xlabel('Posição no Ranking')
    plt.ylabel('Similaridade Média')
    plt.xticks(ranks)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "rank_similarity.png"))
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Teste de Embeddings</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; vertical-align: middle; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h2, h3 {{ color: #333; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .query-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .chart-img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            td img {{ max-width: 150px; height: auto; display: block; margin: 0 auto; border: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Teste de Embeddings</h1>
        
        <h2>Métricas Gerais</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Número de Consultas</h3>
                <div class="metric-value">{num_queries}</div>
            </div>
            <div class="metric-card">
                <h3>Similaridade Média</h3>
                <div class="metric-value">{avg_similarity:.4f}</div>
            </div>
        </div>
        
        <h2>Distribuição de Similaridade</h2>
        <img src="similarity_distribution.png" alt="Distribuição de Similaridade" class="chart-img">
        
        <h2>Similaridade por Posição no Ranking</h2>
        <img src="rank_similarity.png" alt="Similaridade por Rank" class="chart-img">
        
        <h2>Similaridade por Rank</h2>
        {rank_similarity.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)}
        
        <h2>Resultados por Consulta</h2>
    """
    
    for item in query_results:
        html_content += f"""
        <div class="query-section">
            <h3>Consulta: "{item['query']}"</h3>
            <p>Similaridade Média: <strong>{item['avg_similarity']:.4f}</strong></p>
            {item['results_table']}
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(report_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Relatório gerado em {report_html}")

def main():
    parser = argparse.ArgumentParser(description="Teste de embeddings com consultas de exemplo")
    parser.add_argument("--queries", type=str, default="../data/input/test_queries.txt",
                      help="Arquivo com consultas de teste")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Número de resultados por consulta")
    parser.add_argument("--output-dir", type=str, default="../results/embeddings",
                      help="Diretório para salvar resultados")
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        logger.error("API key da OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY.")
        sys.exit(1)
    
    queries = load_test_queries(args.queries)
    
    if not queries:
        logger.error(f"Não foi possível carregar consultas do arquivo {args.queries}")
        sys.exit(1)
    
    logger.info(f"Carregadas {len(queries)} consultas de teste")
    
    run_embedding_tests(queries, args.top_k, args.output_dir)

if __name__ == "__main__":
    main()