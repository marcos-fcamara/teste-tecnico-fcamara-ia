import os
import json
import logging
import pandas as pd
from openai import OpenAI
import random
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key da OpenAI não fornecida")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_queries(self, descriptions_df, num_queries=30):
        """Gera consultas de teste específicas usando a API da OpenAI."""
        sample_rows = descriptions_df.sample(min(10, len(descriptions_df)))
        items_json = sample_rows.to_json(orient='records', force_ascii=False)
        
        prompt = f"""
        Você é um especialista em moda e sistemas de busca de roupas.
        
        Preciso que você gere {num_queries} consultas de busca realistas que um usuário faria para encontrar peças de roupa em um e-commerce.
        
        Aqui estão alguns exemplos de itens da nossa base de dados:
        {items_json}
        
        Gere consultas variadas incluindo:
        1. Consultas simples (ex: "vestido azul")
        2. Consultas específicas (ex: "vestido de festa vermelho com rendas")
        3. Consultas por estilo (ex: "roupa casual verão estilo boho")
        4. Consultas por ocasião (ex: "roupa para trabalho em escritório")
        5. Consultas por composição (ex: "conjunto de saia e blusa para primavera")
        
        Formate a resposta como um array JSON com objetos contendo:
        - "query": a consulta do usuário
        - "categoria": o tipo de consulta (simples, específica, estilo, ocasião, composição)
        - "intenção": o que o usuário está buscando
        
        Importante: Crie consultas bem específicas e variadas que funcionariam bem para testes de sistemas de busca semântica.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em moda e e-commerce."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            try:
                queries_data = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Erro ao decodificar JSON: {content[:100]}...")
                return []
            
            if isinstance(queries_data, dict) and "queries" in queries_data:
                queries = queries_data["queries"]
            elif isinstance(queries_data, list):
                queries = queries_data
            else:
                logger.warning("Formato inesperado na resposta. Tentando extrair consultas...")
                queries = []
                for key, value in queries_data.items():
                    if isinstance(value, dict) and "query" in value:
                        queries.append(value)
                    elif isinstance(value, str):
                        queries.append({"query": value, "categoria": "desconhecida", "intenção": "desconhecida"})
                
            return queries
            
        except Exception as e:
            logger.error(f"Erro ao gerar consultas com a API: {str(e)}")
            return []
    
    def generate_specialized_queries(self, descriptions_df, num_queries=5, category_type="ocasião"):
        """Gera consultas especializadas para uma categoria específica."""
        if category_type == "ocasião":
            category_col = "ocasiao"
        elif category_type == "tipo":
            category_col = "tipo_peca"
        elif category_type == "estilo":
            category_col = "estilo"
        elif category_type == "estação":
            category_col = "estacao"
        else:
            category_col = "cores"
        
        values = set()
        for val in descriptions_df[category_col].dropna():
            if isinstance(val, str):
                values.add(val)
            elif isinstance(val, list):
                for v in val:
                    values.add(v)
        
        values = list(values)
        if not values:
            return []
        
        sample_values = random.sample(values, min(5, len(values)))
        prompt = f"""
        Você é um especialista em moda e sistemas de busca de roupas.
        
        Preciso que você gere {num_queries} consultas de busca realistas específicas para a categoria: {category_type}.
        
        Exemplos de valores dessa categoria em nossa base: {', '.join(sample_values)}
        
        Crie consultas detalhadas e específicas que um usuário usaria para encontrar roupas com essas características.
        As consultas devem ser variadas e focadas em {category_type}.
        
        Formate a resposta como um array JSON com objetos contendo:
        - "query": a consulta do usuário
        - "categoria": o valor do {category_type} alvo
        - "intenção": o que o usuário está buscando
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em moda e e-commerce."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            queries_data = json.loads(content)
            
            if isinstance(queries_data, dict) and "queries" in queries_data:
                return queries_data["queries"]
            elif isinstance(queries_data, list):
                return queries_data
            else:
                logger.warning("Formato inesperado na resposta. Tentando extrair consultas...")
                queries = []
                for key, value in queries_data.items():
                    if isinstance(value, dict) and "query" in value:
                        queries.append(value)
                    elif isinstance(value, str):
                        queries.append({"query": value, "categoria": sample_values[0], "intenção": "desconhecida"})
                
                return queries
            
        except Exception as e:
            logger.error(f"Erro ao gerar consultas especializadas: {str(e)}")
            return []

def main():
    descriptions_file = "test_descriptions.csv"
    output_file = "ai_generated_queries.json"
    text_output_file = "test_queries.txt"
    
    if not os.path.exists(descriptions_file):
        logger.error(f"Arquivo de descrições não encontrado: {descriptions_file}")
        return
    
    try:
        df = pd.read_csv(descriptions_file)
        logger.info(f"Carregadas {len(df)} descrições de {descriptions_file}")
    except Exception as e:
        logger.error(f"Erro ao carregar descrições: {str(e)}")
        return
    
    try:
        generator = QueryGenerator()
    except ValueError as e:
        logger.error(str(e))
        return
    
    all_queries = []
    
    logger.info("Gerando consultas gerais...")
    general_queries = generator.generate_queries(df, num_queries=15)
    all_queries.extend(general_queries)
    
    categories = [
        ("tipo", "tipo_peca", 3),
        ("estilo", "estilo", 3),
        ("ocasião", "ocasiao", 3),
        ("estação", "estacao", 3),
        ("cor", "cores", 3)
    ]
    
    for name, col, count in tqdm(categories, desc="Gerando consultas por categoria"):
        logger.info(f"Gerando consultas por {name}...")
        specialized_queries = generator.generate_specialized_queries(
            df, num_queries=count, category_type=name
        )
        all_queries.extend(specialized_queries)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_queries, f, ensure_ascii=False, indent=2)
        logger.info(f"Salvas {len(all_queries)} consultas em {output_file}")
        
        with open(text_output_file, 'w', encoding='utf-8') as f:
            for i, query_obj in enumerate(all_queries, 1):
                query = query_obj.get("query", f"Consulta {i}")
                f.write(f"{i}. {query}\n")
        logger.info(f"Consultas em formato texto salvas em {text_output_file}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar consultas: {str(e)}")

if __name__ == "__main__":
    main()