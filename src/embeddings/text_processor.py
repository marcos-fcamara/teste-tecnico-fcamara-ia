import os
import logging
from typing import Dict, List, Any, Optional

from openai import OpenAI

from src.utils.cache_manager import CacheManager, cached_embedding

logger = logging.getLogger(__name__)

class TextProcessor:
    """Classe para processar consultas textuais e convertê-las em embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        print("Inicializando TextProcessor")
        print(f"Método enhance_query existe: {hasattr(self, 'enhance_query')}")
        print(f"Método enhance_query é callable: {callable(getattr(self, 'enhance_query', None))}")

        """
        Inicializa o processador de texto.

        Args:
            api_key: Chave da API da OpenAI. Se None, será usada a variável de ambiente.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key da OpenAI não fornecida")

        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        self.cache_manager = CacheManager()

    def enhance_query(self, query_text: str) -> str:
        """
        Aprimora a consulta do usuário com técnicas avançadas para maximizar a similaridade.

        Args:
            query_text: Texto da consulta original.

        Returns:
            str: Consulta aprimorada.
        """
        try:
            cached_query = self.cache_manager.get_cached_query_results(
                f"enhance_v2_{query_text}", 1
            )
            if cached_query:
                logger.info(f"Usando consulta aprimorada em cache")
                return cached_query.get("enhanced_query", query_text)

            logger.info(f"Aprimorando consulta avançada: '{query_text}'")

            extraction_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um especialista em moda com foco em análise de consultas. "
                            "Sua tarefa é analisar uma consulta de busca de roupas e extrair todos os atributos "
                            "relevantes, incluindo atributos implícitos. Forneça sua resposta no formato JSON "
                            "com os seguintes campos:\n"
                            "- tipo_peca: o tipo específico de roupa (ex: vestido, camisa, calça)\n"
                            "- cores: lista de cores mencionadas ou implícitas\n"
                            "- padrao: padrão ou estampa (ex: liso, listrado, floral)\n"
                            "- material: material da peça (ex: algodão, seda, jeans)\n"
                            "- estilo: estilo da peça (ex: casual, formal, esportivo)\n"
                            "- ocasiao: ocasião de uso (ex: festa, trabalho, dia a dia)\n"
                            "- genero: gênero da peça (masculino, feminino, unissex)\n"
                            "- estacao: estação do ano (verão, inverno, etc.)\n"
                            "- caracteristicas: outras características importantes\n\n"
                            "Para campos não mencionados explicitamente, infira a partir do contexto "
                            "ou use null se não for possível inferir."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Analise esta consulta de busca de roupas e extraia todos os atributos relevantes: '{query_text}'",
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            import json

            attributes = json.loads(extraction_response.choices[0].message.content)

            expansion_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um especialista avançado em moda e sistemas de busca semântica. "
                            "Sua tarefa é expandir uma consulta de busca de roupas para maximizar "
                            "a similaridade vetorial. Use os atributos extraídos para criar uma descrição "
                            "extremamente detalhada e rica, abordando cada aspecto da peça.\n\n"
                            "Use linguagem natural e fluidez, intercalando sinônimos e termos alternativos "
                            "para cada atributo importante. Inclua frases que combinem múltiplos atributos. "
                            "Mencione cada cor em conjunto com elementos como tonalidade, intensidade e combinações.\n\n"
                            "Use linguagem rica, precisa e técnica do domínio da moda. Cada atributo deve ser "
                            "mencionado várias vezes em diferentes contextos. Inclua elementos visuais detalhados "
                            "que ajudem a distinguir a peça.\n\n"
                            "Sua resposta deve ser extremamente rica, com alta correspondência semântica, "
                            "mantendo a essência original da consulta."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""Consulta original: '{query_text}'

            Atributos extraídos:
            {json.dumps(attributes, indent=2, ensure_ascii=False)}

            Crie uma descrição expandida para maximizar a similaridade vetorial:""",
                    },
                ],
                temperature=0.3,
                max_tokens=500,
            )

            enhanced_query = expansion_response.choices[0].message.content

            structured_query = f"""Consulta Detalhada de Moda: {enhanced_query}

            Principais Atributos:
            - Tipo de Peça: {attributes.get('tipo_peca', 'N/A')}
            - Cores: {', '.join(attributes.get('cores', [])) if isinstance(attributes.get('cores'), list) else attributes.get('cores', 'N/A')}
            - Padrão/Estampa: {attributes.get('padrao', 'N/A')}
            - Material: {attributes.get('material', 'N/A')}
            - Estilo: {attributes.get('estilo', 'N/A')}
            - Ocasião: {attributes.get('ocasiao', 'N/A')}
            - Gênero: {attributes.get('genero', 'N/A')}
            - Estação: {attributes.get('estacao', 'N/A')}

            {enhanced_query}

            Termos Relevantes: {query_text}, {attributes.get('tipo_peca', '')}, {attributes.get('estilo', '')}, {attributes.get('ocasiao', '')}"""

            logger.debug(f"Consulta aprimorada avançada: '{structured_query[:100]}...'")

            self.cache_manager.cache_query_results(
                f"enhance_v2_{query_text}", 1, {"enhanced_query": structured_query}
            )

            return structured_query

        except Exception as e:
            logger.warning(
                f"Erro ao aprimorar consulta avançada: {str(e)}. Usando consulta original."
            )
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um especialista em moda. Expanda esta consulta de busca de roupas com detalhes "
                                "sobre tipo de peça, cores, padrões, materiais, estilo, ocasião, gênero e estação."
                            ),
                        },
                        {"role": "user", "content": f"Expanda: '{query_text}'"},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )

                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Erro no enhance_query: {e}. Usando consulta original.")
                return query_text.strip('"')


    def _preprocess_text(self, text: str) -> str:
        import re

        text = re.sub(r"\s+", " ", text.strip())

        tipos_peca = [
            "vestido",
            "camisa",
            "calça",
            "casaco",
            "jaqueta",
            "saia",
            "blusa",
            "terno",
            "camiseta",
        ]
        for tipo in tipos_peca:
            text = re.sub(
                r"\b(" + tipo + r")\b", r"<TIPO>\1</TIPO>", text, flags=re.IGNORECASE
            )

        cores = [
            "vermelho",
            "azul",
            "verde",
            "preto",
            "branco",
            "amarelo",
            "rosa",
            "laranja",
            "roxo",
            "marrom",
            "cinza",
        ]
        for cor in cores:
            text = re.sub(
                r"\b(" + cor + r")\b", r"<COR>\1</COR>", text, flags=re.IGNORECASE
            )

        estilos = [
            "formal",
            "casual",
            "esportivo",
            "elegante",
            "vintage",
            "moderno",
            "clássico",
            "fashion",
        ]
        for estilo in estilos:
            text = re.sub(
                r"\b(" + estilo + r")\b", r"<ESTILO>\1</ESTILO>", text, flags=re.IGNORECASE
            )

        abreviacoes = {
            "p/": "para",
            "c/": "com",
            "s/": "sem",
            "tbm": "também",
            "p": "pequeno",
            "m": "médio",
            "g": "grande",
            "gg": "extra grande",
        }
        for abrev, expansao in abreviacoes.items():
            text = re.sub(r"\b" + abrev + r"\b", expansao, text, flags=re.IGNORECASE)

        return text


    @cached_embedding(CacheManager())
    def generate_embedding(self, text: str, use_ensemble: bool = True, force_local: bool = False) -> List[float]:
        try:
            embedding_key = f"{text}_ensemble" if use_ensemble else text
            cached_embedding = self.cache_manager.get_cached_embedding(embedding_key)
            if cached_embedding:
                logger.info(f"Usando embedding em cache")
                return cached_embedding

            if force_local:
                logger.info(f"Modo force_local ativado, mas embedding não encontrado no cache")
                return None

            logger.info(f"Gerando embedding avançado para texto")

            if use_ensemble:
                processed_text = self._preprocess_text(text)

                embeddings = []

                response = self.client.embeddings.create(
                    model=self.embedding_model, input=text
                )
                base_embedding = response.data[0].embedding
                embeddings.append(base_embedding)

                response = self.client.embeddings.create(
                    model=self.embedding_model, input=processed_text
                )
                proc_embedding = response.data[0].embedding
                embeddings.append(proc_embedding)

                import re

                tipos = re.findall(
                    r"\b(vestido|camisa|calça|casaco|jaqueta|saia|blusa|terno|camiseta)\b",
                    text,
                    re.IGNORECASE,
                )
                cores = re.findall(
                    r"\b(vermelho|azul|verde|preto|branco|amarelo|rosa|laranja|roxo|marrom|cinza)\b",
                    text,
                    re.IGNORECASE,
                )
                estilos = re.findall(
                    r"\b(formal|casual|esportivo|elegante|vintage|moderno|clássico)\b",
                    text,
                    re.IGNORECASE,
                )
                padroes = re.findall(
                    r"\b(liso|estampado|floral|listrado|xadrez)\b", text, re.IGNORECASE
                )

                structured_text = text
                if tipos:
                    structured_text += f"\nTipo: {', '.join(tipos)}"
                if cores:
                    structured_text += f"\nCores: {', '.join(cores)}"
                if estilos:
                    structured_text += f"\nEstilo: {', '.join(estilos)}"
                if padroes:
                    structured_text += f"\nPadrão: {', '.join(padroes)}"

                response = self.client.embeddings.create(
                    model=self.embedding_model, input=structured_text
                )
                struct_embedding = response.data[0].embedding
                embeddings.append(struct_embedding)

                weights = [0.5, 0.2, 0.3]

                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                combined_embedding = []
                for i in range(len(base_embedding)):
                    weighted_sum = sum(
                        embedding[i] * weight
                        for embedding, weight in zip(embeddings, weights)
                    )
                    combined_embedding.append(weighted_sum)

                import numpy as np

                norm = np.linalg.norm(combined_embedding)
                if norm > 0:
                    final_embedding = [float(val / norm) for val in combined_embedding]
                else:
                    final_embedding = combined_embedding

                self.cache_manager.cache_embedding(embedding_key, final_embedding)

                return final_embedding
            else:
                response = self.client.embeddings.create(
                    model=self.embedding_model, input=text
                )

                embedding = response.data[0].embedding

                self.cache_manager.cache_embedding(embedding_key, embedding)

                return embedding

        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            if force_local:
                return None
            raise
