import os
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_queries_with_openai(categories, api_key=None):
    """Gera consultas utilizando a API OpenAI"""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key da OpenAI não fornecida")
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Crie 25 consultas específicas de busca para um sistema de roupas, baseadas nestas categorias:
    
    Tipos de peças: {', '.join(categories.get('tipos', []))}
    Cores: {', '.join(categories.get('cores', []))}
    Estilos: {', '.join(categories.get('estilos', []))}
    Ocasiões: {', '.join(categories.get('ocasioes', []))}
    Estações: {', '.join(categories.get('estacoes', []))}
    
    As consultas devem ser realistas e específicas, como um cliente faria em um e-commerce.
    Inclua consultas variadas combinando diferentes atributos.
    
    Formate as consultas como uma lista numerada, uma por linha.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em moda e e-commerce."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        lines = content.strip().split('\n')
        
        queries = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                query = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                queries.append(query)
        
        return queries
        
    except Exception as e:
        logger.error(f"Erro na chamada da API: {str(e)}")
        return []

def extract_categories(df):
    """Extrai categorias das descrições para usar nas consultas"""
    categories = {
        'tipos': [],
        'cores': [],
        'estilos': [],
        'ocasioes': [],
        'estacoes': []
    }
    
    for col, cat_key in [
        ('tipo_peca', 'tipos'),
        ('cores', 'cores'),
        ('estilo', 'estilos'),
        ('ocasiao', 'ocasioes'),
        ('estacao', 'estacoes')
    ]:
        values = set()
        for val in df[col].dropna():
            if isinstance(val, str):
                values.add(val)
            elif isinstance(val, list):
                for v in val:
                    values.add(v)
        
        categories[cat_key] = list(values)[:5]
    
    return categories

def main():
    descriptions_file = "test_descriptions.csv"
    output_file = "test_queries.txt"
    
    try:
        df = pd.read_csv(descriptions_file)
        logger.info(f"Carregadas {len(df)} descrições")
    except Exception as e:
        logger.error(f"Erro ao carregar descrições: {str(e)}")
        return
    
    categories = extract_categories(df)
    
    queries = generate_queries_with_openai(categories)
    
    if not queries:
        logger.warning("Nenhuma consulta gerada. Usando consultas padrão.")
        queries = [
            "Vestido floral para verão",
            "Macaquinho preto elegante",
            "Conjunto de top e short casual",
            "Vestido branco rendado para festa",
            "Roupa azul para o dia a dia",
            "Peça de roupa estilo boho",
            "Vestido para ocasião formal",
            "Macaquinho estampado para praia",
            "Vestido casual para primavera",
            "Roupa com padrão listrado"
        ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, query in enumerate(queries, 1):
            f.write(f"{i}. {query}\n")
    
    logger.info(f"Salvas {len(queries)} consultas em {output_file}")

if __name__ == "__main__":
    main()