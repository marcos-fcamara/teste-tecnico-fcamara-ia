#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar consultas de buscas e gerar um relatório completo com visualizações.
"""

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
from datetime import datetime

# Adicionar diretório raiz ao path do Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
os.environ["CHROMA_PERSIST_DIRECTORY"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chroma_db'))

# Diretório de saída padrão
output_dir = "../results/search_report"

# Tentar importar módulos com tratamento de erro explícito
try:
    from src.processing.indexer import ImageIndexer
    from src.processing.image_processor import ImageProcessor
    from src.database.vector_db import VectorDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Verifique se o diretório raiz do projeto está no PYTHONPATH")
    sys.exit(1)

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lista de consultas de teste padrão
DEFAULT_TEST_QUERIES = [
    "Vestido boho estampado: Vestido curto, solto, estampado em azul e verde, mangas três quartos, decote V, estilo praiano.",
    "Vestido preto de renda: Vestido elegante, preto, solto, mangas transparentes de renda com bolinhas, gola dobrável."
    "Conjunto boho casual: Top curto vazado e shorts de cintura alta, branco ou floral, com renda e franjas."
    "Vestido branco sexy: Conjunto de top de renda floral e saia longa com fenda, elegante e fresco.",
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
    "roupa com estampa animal para mulher",
    "vestido vermelho elegante para ocasião formal"
]

def load_test_queries(queries_file: str = None) -> List[str]:
    """Carrega consultas de teste de um arquivo ou usa consultas padrão."""
    if not queries_file:
        logger.info("Nenhum arquivo de consultas especificado. Usando consultas padrão.")
        return DEFAULT_TEST_QUERIES
    
    try:
        # Tentar com o caminho fornecido
        if os.path.exists(queries_file):
            logger.info(f"Carregando consultas de: {queries_file}")
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
                # Remover numeração se presente (ex: "1. query" -> "query")
                queries = [re.sub(r'^\d+\.\s+', '', q) for q in queries]
            return queries
        
        # Tentar com caminho absoluto alternativo
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        alt_path = os.path.join(base_dir, 'src', 'data', 'input', 'test_queries.txt')
        if os.path.exists(alt_path):
            logger.info(f"Carregando consultas de caminho alternativo: {alt_path}")
            with open(alt_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
                # Remover numeração se presente
                queries = [re.sub(r'^\d+\.\s+', '', q) for q in queries]
            return queries
        
        # Se não encontrar, usar consultas padrão
        logger.warning(f"Arquivo de consultas não encontrado: {queries_file}")
        logger.info("Usando consultas padrão")
        return DEFAULT_TEST_QUERIES
    
    except Exception as e:
        logger.error(f"Erro ao carregar consultas: {str(e)}")
        return DEFAULT_TEST_QUERIES

def run_search_tests(queries: List[str], top_k: int = 3, output_dir: str = "../results/search_report", reuse_embeddings: bool = False):
    """Executa testes de busca e gera relatório completo."""
    if not queries:
        logger.error("Nenhuma consulta para testar.")
        return
    
    # Normalizar e criar o diretório de saída
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Diretório de saída: {output_dir}")
    
    # Criar diretório para imagens no relatório
    images_report_dir = os.path.join(output_dir, "images")
    os.makedirs(images_report_dir, exist_ok=True)
    
    # Configurar os componentes
    vector_db = VectorDatabase()
    image_processor = ImageProcessor()
    
    indexer = ImageIndexer(
        image_processor=image_processor,
        vector_db=vector_db
    )
    
    # Verificar se o modo de reuso está ativado
    if reuse_embeddings:
        logger.info("Modo reuse_embeddings ativado! Usando apenas embeddings em cache.")
    
    # Verificar o banco de dados
    db_info = vector_db.get_collection_info()
    if db_info["item_count"] == 0:
        logger.error("O banco de dados está vazio. Execute a indexação primeiro.")
        sys.exit(1)
    
    logger.info(f"Banco de dados contém {db_info['item_count']} itens indexados.")
    
    # Lista para armazenar todos os resultados
    results_data = []
    
    # Testar cada consulta
    for query in tqdm(queries, desc="Testando consultas"):
        logger.info(f"Testando consulta: '{query}'")
        
        try:
            search_results = indexer.search_by_text(
                query, 
                limit=top_k,
                reuse_embeddings=reuse_embeddings
            )
            
            if "error" in search_results:
                logger.error(f"Erro na consulta '{query}': {search_results['error']}")
                continue
            
            raw_results = search_results["results"]
            
            for i in range(len(raw_results["ids"][0])):
                # Extrair dados do resultado
                image_id = raw_results["ids"][0][i]
                distance = raw_results["distances"][0][i]
                similarity = 1 - distance
                
                metadata = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
                document = raw_results["documents"][0][i] if raw_results["documents"] else ""
                
                # Extrair informações usando regex - abordagem robusta
                tipo_peca = "N/A"
                cores = "N/A"
                image_path = "N/A"
                
                # Obter caminho da imagem
                if metadata and "path" in metadata:
                    image_path = metadata["path"]
                    # Extrair filename
                    filename = os.path.basename(image_path)
                else:
                    # Tentar extrair do documento
                    image_path_match = re.search(r'"path"\s*:\s*"([^"]+)"', document) or re.search(r'"filename"\s*:\s*"([^"]+)"', document)
                    if image_path_match:
                        image_path = image_path_match.group(1)
                        filename = os.path.basename(image_path)
                    else:
                        filename = f"imagem_{image_id}.jpg"
                
                # Extrair tipo de peça
                tipo_peca_match = re.search(r'"tipo_de_peca"\s*:\s*"([^"]+)"', document) or re.search(r'"tipo_peca"\s*:\s*"([^"]+)"', document)
                if tipo_peca_match:
                    tipo_peca = tipo_peca_match.group(1)
                
                # Extrair cores - pode ser lista ou string
                cores_match = re.search(r'"cores_predominantes"\s*:\s*(\[[^\]]+\])', document) or re.search(r'"cores"\s*:\s*(\[[^\]]+\])', document)
                if cores_match:
                    try:
                        cores_array = json.loads(cores_match.group(1))
                        if isinstance(cores_array, list):
                            cores = ", ".join(cores_array)
                    except:
                        cores_text = cores_match.group(1).strip('[]').replace('"', '')
                        cores = cores_text
                
                # Se ainda não tiver cores, procurar por versão string
                if cores == "N/A":
                    cores_str_match = re.search(r'"cores_predominantes"\s*:\s*"([^"]+)"', document) or re.search(r'"cores"\s*:\s*"([^"]+)"', document)
                    if cores_str_match:
                        cores = cores_str_match.group(1)
                
                # Adicionar aos resultados
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
        
        except Exception as e:
            logger.error(f"Erro ao processar consulta '{query}': {str(e)}")
            continue
    
    # Converter para DataFrame
    df = pd.DataFrame(results_data)
    
    if df.empty:
        logger.error("Nenhum resultado encontrado para as consultas testadas.")
        return
    
    # Salvar resultados em CSV
    output_csv = os.path.join(output_dir, "search_test_results.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Resultados salvos em {output_csv}")
    
    # Gerar relatório tabular com tabulate
    headers = ["Consulta", "Rank", "Imagem", "Similaridade", "Tipo de Peça", "Cores"]
    tabular_data = df[["query", "rank", "filename", "similarity", "tipo_peca", "cores"]].values.tolist()
    
    output_txt = os.path.join(output_dir, "search_test_results.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Resultados das Consultas de Busca\n\n")
        f.write(tabulate(tabular_data, headers=headers, tablefmt="grid"))
    
    logger.info(f"Relatório tabular salvo em {output_txt}")
    
    # Gerar relatório HTML com visualizações
    generate_search_report(df, output_dir)
    
    return df

def generate_search_report(results_df: pd.DataFrame, output_dir: str):
    """Gera relatório HTML com visualizações a partir dos resultados."""
    if results_df.empty:
        logger.error("Sem dados para gerar relatório.")
        return
    
    # Extrair métricas gerais
    num_queries = results_df['query'].nunique()
    avg_similarity = results_df['similarity'].mean()
    
    # Calcular estatísticas por rank
    rank_similarity = results_df.groupby('rank')['similarity'].agg(['mean', 'min', 'max']).reset_index()
    
    # Caminho para o relatório HTML
    report_html = os.path.join(output_dir, "search_test_report.html")
    
    # Criar diretório para imagens no relatório
    images_report_dir = os.path.join(output_dir, "images")
    os.makedirs(images_report_dir, exist_ok=True)
    
    # Copiar todas as imagens para o diretório do relatório
    # Copiar todas as imagens para o diretório do relatório
    for _, row in results_df.iterrows():
        # Lista de possíveis localizações para procurar a imagem
        img_basename = os.path.basename(row['image_path'])
        possible_paths = [
            row['image_path'],  # Caminho original do metadata
            os.path.join('/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/src/data/images', img_basename),
            os.path.join(os.path.abspath('../../src/data/images'), img_basename),
            os.path.join(os.path.abspath('../../../src/data/images'), img_basename),
            os.path.join(os.getcwd(), 'src/data/images', img_basename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/data/images', img_basename)
        ]
        
        # Verificar cada caminho possível
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    dest_path = os.path.join(images_report_dir, img_basename)
                    shutil.copy2(path, dest_path)
                    logger.info(f"Imagem copiada: {path} -> {dest_path}")
                    found = True
                    break  # Encontrou e copiou, não precisa continuar procurando
                except Exception as e:
                    logger.error(f"Erro ao copiar {path}: {e}")
        
        # Se não encontrou a imagem em nenhum caminho possível, cria um placeholder
        if not found:
            logger.warning(f"Imagem não encontrada em nenhum caminho possível: {img_basename}")
            try:
                from PIL import Image, ImageDraw
                
                # Cria uma imagem placeholder
                placeholder = Image.new('RGB', (220, 220), color=(240, 240, 240))
                draw = ImageDraw.Draw(placeholder)
                
                # Adiciona texto informativo
                text = f"Imagem não encontrada"
                draw.text((30, 100), text, fill=(0, 0, 0))
                draw.text((30, 120), img_basename[:20], fill=(0, 0, 0))
                
                # Salva o placeholder no diretório de imagens
                dest_path = os.path.join(images_report_dir, img_basename)
                placeholder.save(dest_path)
                logger.info(f"Placeholder criado para: {img_basename}")
            except Exception as e:
                logger.error(f"Erro ao criar placeholder: {str(e)}")
    
    # Preparar dados para o relatório por consulta
    query_results = []
    for query in results_df['query'].unique():
        query_df = results_df[results_df['query'] == query].sort_values('rank')
        
        # Criar coluna de imagem HTML
        image_html = []
        for _, row in query_df.iterrows():
            img_basename = os.path.basename(row['image_path'])
            # Verificar se a imagem existe no diretório do relatório
            img_report_path = os.path.join(images_report_dir, img_basename)
            if os.path.exists(img_report_path):
                img_tag = f'<img src="images/{img_basename}" alt="{row["tipo_peca"]}" width="150">'
            else:
                # Se não existe, usar um placeholder HTML
                img_tag = f'''<div style="width:150px;height:150px;background:#f0f0f0;display:flex;
                align-items:center;justify-content:center;margin:0 auto;border:1px solid #ddd;">
                <div style="text-align:center;font-size:12px;padding:5px;">
                Imagem não encontrada<br/>{img_basename[:15]}...</div></div>'''
            image_html.append(img_tag)
        
        # Adicionar coluna de imagem HTML
        temp_df = query_df.copy()
        temp_df['imagem'] = image_html
        
        # Gerar tabela HTML
        query_table = temp_df[['rank', 'imagem', 'similarity', 'tipo_peca', 'cores']].to_html(
            index=False,
            escape=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        
        query_results.append({
            "query": query,
            "avg_similarity": query_df['similarity'].mean(),
            "results_table": query_table
        })
    
    # Gerar gráficos
    # 1. Distribuição de similaridade
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['similarity'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribuição de Similaridade')
    plt.xlabel('Similaridade')
    plt.ylabel('Frequência')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
    
    # 2. Similaridade média por rank
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
    
    # 3. Top 5 consultas com maior similaridade média
    top_queries = results_df.groupby('query')['similarity'].mean().nlargest(5).reset_index()
    plt.figure(figsize=(12, 6))
    plt.barh(top_queries['query'], top_queries['similarity'], color='teal', alpha=0.7)
    plt.title('Top 5 Consultas com Maior Similaridade Média')
    plt.xlabel('Similaridade Média')
    plt.ylabel('Consulta')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_queries.png"))
    
    # Criar HTML com todos os elementos
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Testes de Busca</title>
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
        <h1>Relatório de Testes de Busca</h1>
        
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
            <div class="metric-card">
                <h3>Total de Resultados</h3>
                <div class="metric-value">{len(results_df)}</div>
            </div>
        </div>
        
        <h2>Distribuição de Similaridade</h2>
        <img src="similarity_distribution.png" alt="Distribuição de Similaridade" class="chart-img">
        
        <h2>Similaridade por Posição no Ranking</h2>
        <img src="rank_similarity.png" alt="Similaridade por Rank" class="chart-img">
        
        <h2>Top 5 Consultas com Maior Similaridade</h2>
        <img src="top_queries.png" alt="Top 5 Consultas" class="chart-img">
        
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
    
    logger.info(f"Relatório HTML gerado em {report_html}")

def main():
    parser = argparse.ArgumentParser(description="Testes de busca com geração de relatório completo")
    parser.add_argument("--queries", type=str, default=None,
                      help="Arquivo com consultas de teste (opcional)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Número de resultados por consulta")
    parser.add_argument("--output-dir", type=str, default="../results/search_report",
                      help="Diretório para salvar resultados e relatórios")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Reutiliza embeddings existentes sem recalcular")
    
    args = parser.parse_args()
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("API key da OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY.")
        sys.exit(1)
    
    # Carregar consultas
    queries = load_test_queries(args.queries)
    
    logger.info(f"Carregadas {len(queries)} consultas de teste")
    
    # Executar testes
    run_search_tests(
        queries=queries, 
        top_k=args.top_k, 
        output_dir=args.output_dir,
        reuse_embeddings=args.reuse_embeddings
    )
    
    logger.info("Testes de busca concluídos com sucesso!")

if __name__ == "__main__":
    main()