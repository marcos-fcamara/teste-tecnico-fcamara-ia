import os
import sys
import logging
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','src')))

# Importa os componentes do sistema
from src.processing.indexer import ImageIndexer
from src.processing.image_processor import ImageProcessor
from src.database.vector_db import VectorDatabase
from src.embeddings.text_processor import TextProcessor


def run_similarity_test(queries: List[str], output_dir: str = "../results/similarity"):
    """
    Executa testes de similaridade com diferentes técnicas.

    Args:
        queries: Lista de consultas para testar.
        output_dir: Diretório para salvar os resultados.
    """
    # Carrega variáveis de ambiente
    load_dotenv()

    # Verifica se a API key da OpenAI está configurada
    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "API key da OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY."
        )
        sys.exit(1)

    # Cria diretório para resultados
    os.makedirs(output_dir, exist_ok=True)

    # Inicializa os componentes
    image_processor = ImageProcessor()
    vector_db = VectorDatabase()
    text_processor = TextProcessor()

    indexer = ImageIndexer(image_processor=image_processor, vector_db=vector_db)

    # Resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"similarity_test_{timestamp}.json")
    results_plot = os.path.join(output_dir, f"similarity_comparison_{timestamp}.png")

    all_results = {}

    # Executa as consultas
    for query in queries:
        logger.info(f"Testando consulta: '{query}'")

        try:
            # Técnica padrão
            logger.info("  Executando consulta com técnica padrão...")
            standard_result = indexer.search_by_text(query, limit=5, high_quality=False)

            # Técnica avançada
            logger.info("  Executando consulta com técnica avançada...")
            advanced_result = indexer.search_by_text(query, limit=5, high_quality=True)

            # Extrai similaridades
            def extract_similarities(result):
                if "error" in result:
                    return []

                distances = result["results"]["distances"][0]
                return [round((1 - dist) * 100, 2) for dist in distances]

            standard_similarities = extract_similarities(standard_result)
            advanced_similarities = extract_similarities(advanced_result)

            # Registra resultados
            query_results = {
                "standard": {
                    "similarities": standard_similarities,
                    "average": (
                        round(np.mean(standard_similarities), 2)
                        if standard_similarities
                        else 0
                    ),
                    "max": (
                        round(np.max(standard_similarities), 2)
                        if standard_similarities
                        else 0
                    ),
                    "enhanced_query": standard_result.get("enhanced_query", "")[:200]
                    + "...",
                },
                "advanced": {
                    "similarities": advanced_similarities,
                    "average": (
                        round(np.mean(advanced_similarities), 2)
                        if advanced_similarities
                        else 0
                    ),
                    "max": (
                        round(np.max(advanced_similarities), 2)
                        if advanced_similarities
                        else 0
                    ),
                    "enhanced_query": advanced_result.get("enhanced_query", "")[:200]
                    + "...",
                },
            }

            all_results[query] = query_results

            # Mostra resultados
            logger.info(f"  Resultados para '{query}':")
            logger.info(f"    Técnica padrão: {standard_similarities}")
            logger.info(f"      Média: {query_results['standard']['average']}%")
            logger.info(f"      Máxima: {query_results['standard']['max']}%")
            logger.info(f"    Técnica avançada: {advanced_similarities}")
            logger.info(f"      Média: {query_results['advanced']['average']}%")
            logger.info(f"      Máxima: {query_results['advanced']['max']}%")
            logger.info(
                f"    Melhoria: {round(query_results['advanced']['average'] - query_results['standard']['average'], 2)}%"
            )

        except Exception as e:
            logger.error(f"  Erro ao processar consulta '{query}': {str(e)}")
            all_results[query] = {"error": str(e)}

    # Salva resultados
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Resultados salvos em {results_file}")

    # Gera gráfico comparativo
    try:
        plot_results(all_results, results_plot)
        logger.info(f"Gráfico salvo em {results_plot}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico: {str(e)}")


def plot_results(results: Dict[str, Any], output_file: str):
    """
    Gera um gráfico comparativo dos resultados.

    Args:
        results: Resultados dos testes.
        output_file: Arquivo para salvar o gráfico.
    """
    # Prepara dados para o gráfico
    queries = list(results.keys())
    standard_avg = [
        results[q]["standard"]["average"] for q in queries if "error" not in results[q]
    ]
    advanced_avg = [
        results[q]["advanced"]["average"] for q in queries if "error" not in results[q]
    ]
    standard_max = [
        results[q]["standard"]["max"] for q in queries if "error" not in results[q]
    ]
    advanced_max = [
        results[q]["advanced"]["max"] for q in queries if "error" not in results[q]
    ]

    # Remove consultas com erro
    valid_queries = [q for q in queries if "error" not in results[q]]

    # Configura o gráfico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Gráfico de barras para médias
    x = np.arange(len(valid_queries))
    width = 0.35

    ax1.bar(x - width / 2, standard_avg, width, label="Técnica Padrão")
    ax1.bar(x + width / 2, advanced_avg, width, label="Técnica Avançada")

    ax1.set_ylabel("Similaridade Média (%)")
    ax1.set_title("Comparação de Similaridade Média por Consulta")
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_queries, rotation=45, ha="right")
    ax1.legend()

    # Adiciona valores nas barras
    for i, v in enumerate(standard_avg):
        ax1.text(i - width / 2, v + 1, f"{v}%", ha="center")

    for i, v in enumerate(advanced_avg):
        ax1.text(i + width / 2, v + 1, f"{v}%", ha="center")

    # Gráfico de barras para máximos
    ax2.bar(x - width / 2, standard_max, width, label="Técnica Padrão")
    ax2.bar(x + width / 2, advanced_max, width, label="Técnica Avançada")

    ax2.set_ylabel("Similaridade Máxima (%)")
    ax2.set_title("Comparação de Similaridade Máxima por Consulta")
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_queries, rotation=45, ha="right")
    ax2.legend()

    # Adiciona valores nas barras
    for i, v in enumerate(standard_max):
        ax2.text(i - width / 2, v + 1, f"{v}%", ha="center")

    for i, v in enumerate(advanced_max):
        ax2.text(i + width / 2, v + 1, f"{v}%", ha="center")

    # Adiciona linha de 95%
    ax1.axhline(y=95, color="r", linestyle="--", alpha=0.7)
    ax1.text(len(valid_queries) - 1, 96, "Meta: 95%", color="r", ha="right")

    ax2.axhline(y=95, color="r", linestyle="--", alpha=0.7)
    ax2.text(len(valid_queries) - 1, 96, "Meta: 95%", color="r", ha="right")

    # Ajusta layout e salva
    plt.tight_layout()
    plt.savefig(output_file)


def main():
    """Função principal."""
    # Consultas de teste
    queries = [
        "vestido vermelho",
        "camisa social branca",
        "calça jeans azul",
        "roupa casual para verão",
        "vestido floral para festas",
        "camiseta preta básica",
        "roupa formal para homem",
        "blusa feminina para trabalho",
    ]

    run_similarity_test(queries, "../results/similarity")


if __name__ == "__main__":
    main()
