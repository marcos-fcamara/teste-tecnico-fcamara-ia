import os
import json
import glob
import pandas as pd
import logging
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_descriptions(cache_dir="../../src/data/cache/descriptions", output_file="../data/input/test_descriptions.csv", limit=20):
    """
    Extrai descrições de imagens dos arquivos JSON de cache e cria uma base de testes.
    """
    logger.info(f"Buscando arquivos de cache em: {cache_dir}")
    
    json_files = glob.glob(os.path.join(cache_dir, "*.json"))
    logger.info(f"Encontrados {len(json_files)} arquivos JSON")
    
    if not json_files:
        logger.error(f"Nenhum arquivo JSON encontrado em {cache_dir}")
        return None
    
    # Limita a quantidade de arquivos, selecionando aleatoriamente se necessário
    if limit > 0 and len(json_files) > limit:
        import random
        json_files = random.sample(json_files, limit)
        logger.info(f"Limitando para {limit} arquivos aleatórios")
    
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = data.get("id", os.path.basename(json_file))
            # Adicionar caminho da imagem original
            image_path = data.get("image_path", "")
            
            # Correção para o parsing do JSON
            description_raw = data.get("description", "")
            description_text = data.get("description_text", "")
            
            # Tenta extrair o JSON da string se ela começar com ```json
            description = {}
            if isinstance(description_raw, str) and "```json" in description_raw:
                try:
                    # Extrai o conteúdo JSON da string markdown
                    json_text = description_raw.split("```json", 1)[1].split("```", 1)[0].strip()
                    description = json.loads(json_text)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Erro ao extrair JSON da descrição em {json_file}: {str(e)}")
            elif isinstance(description_raw, dict):
                description = description_raw
            
            # Agora use description para extrair os valores
            tipo_peca = description.get("tipo_de_peca", description.get("Tipo de peça", ""))
            ocasiao = description.get("ocasiao_de_uso", description.get("Ocasião de uso", ""))
            cores = description.get("cores_predominantes", description.get("Cores predominantes", ""))
            padrao = description.get("padrao", description.get("Padrão", ""))
            estilo = description.get("estilo_estetica", description.get("Estilo/estética", ""))
            genero = description.get("genero", description.get("Gênero", ""))
            estacao = description.get("estacao_do_ano_mais_adequada", description.get("Estação do ano mais adequada", ""))
            
            # Extrai a descrição completa do JSON interno, se existir
            descricao_completa = description.get("descricao_completa", "")
            if not descricao_completa and description_text:
                descricao_completa = description_text
            
            all_data.append({
                "arquivo": file_name,
                "imagem_original": image_path,
                "tipo_peca": tipo_peca,
                "ocasiao": ocasiao,
                "cores": cores,
                "padrao": padrao,
                "estilo": estilo,
                "genero": genero,
                "estacao": estacao,
                "descricao_completa": descricao_completa
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar {json_file}: {str(e)}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Dados salvos em {output_file}")
        
        table_txt_file = os.path.splitext(output_file)[0] + ".txt"
        with open(table_txt_file, 'w', encoding='utf-8') as f:
            f.write(tabulate(df, headers='keys', tablefmt='grid'))
        logger.info(f"Tabela formatada salva em {table_txt_file}")
        
        table_html_file = os.path.splitext(output_file)[0] + ".html"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Descrições de Imagens</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Base de Testes - Descrições de Imagens</h1>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        with open(table_html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Tabela HTML salva em {table_html_file}")
        
        return df
    else:
        logger.warning("Nenhum dado foi processado.")
        return None

def generate_test_queries(df, num_queries=30, output_file="../data/input/test_queries.txt"):
    """
    Gera consultas de teste com base nas descrições extraídas.
    """
    if df is None or df.empty:
        logger.error("DataFrame vazio ou nulo. Não é possível gerar consultas.")
        return
    
    queries = []
    
    tipos_pecas = df['tipo_peca'].dropna().unique()
    for tipo in tipos_pecas[:min(3, len(tipos_pecas))]:
        queries.append(f"{tipo} para uso casual")
    
    cores = set()
    for cor_item in df['cores'].dropna():
        if isinstance(cor_item, str):
            for cor in cor_item.split(','):
                cor = cor.strip()
                if cor:
                    cores.add(cor)
        elif isinstance(cor_item, list):
            for cor in cor_item:
                cor = cor.strip()
                if cor:
                    cores.add(cor)
    
    for cor in list(cores)[:min(3, len(cores))]:
        queries.append(f"Roupa {cor} para o dia a dia")
    
    estilos = df['estilo'].dropna().unique()
    for estilo in estilos[:min(2, len(estilos))]:
        queries.append(f"Peça de roupa com estilo {estilo}")
    
    estacoes = df['estacao'].dropna().unique()
    for estacao in estacoes[:min(2, len(estacoes))]:
        queries.append(f"Roupa para {estacao}")
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        if pd.notna(row['tipo_peca']):
            if isinstance(row['cores'], list) and len(row['cores']) > 0:
                cor = row['cores'][0]
                queries.append(f"{row['tipo_peca']} {cor} {row.get('estilo', '')}")
            elif pd.notna(row['cores']) and isinstance(row['cores'], str):
                cor = row['cores'].split(',')[0].strip()
                queries.append(f"{row['tipo_peca']} {cor} {row.get('estilo', '')}")
    
    queries = queries[:num_queries]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, query in enumerate(queries, 1):
            f.write(f"{i}. {query}\n")
    
    logger.info(f"Geradas {len(queries)} consultas de teste em {output_file}")
    return queries

if __name__ == "__main__":
    df = extract_descriptions(limit=20)
    
    if df is not None:
        generate_test_queries(df, output_file="../data/input/test_queries.txt")