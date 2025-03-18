# Teste Técnico fCamara - Engenheiro de IA

Sistema cloud-native que permite busca em linguagem natural por itens de moda, aproveitando os modelos de visão e embedding da OpenAI para entender e combinar descrições de roupas. O sistema processa imagens de moda, gera descrições detalhadas e permite busca semântica através de embeddings vetoriais.

## Principais Características

- Pipeline de normalização de imagem para qualidade consistente
- Descrições detalhadas de itens de moda usando GPT-4V
- Busca semântica usando embeddings de texto com pesos personalizados
- API RESTful para operações de busca
- Armazenamento em banco de dados vetorial escalável usando ChromaDB
- Ferramentas abrangentes de teste e avaliação
- Sistema de cache eficiente para descrições e embeddings

## Estrutura do Repositório

```
teste-tecnico-fcamara/
├── docs/                      # Diagramas de arquitetura e componentes
├── src/                      # Código-fonte principal da aplicação
│   ├── api/                 # Aplicação FastAPI para endpoints de busca
│   ├── database/            # Implementação de banco de dados vetorial usando ChromaDB
│   └── processing/          # Componentes de processamento de imagem e texto
│       ├── image_normalizer.py  # Melhoria de qualidade de imagem
│       ├── image_processor.py   # Geração de descrição de imagem
│       └── indexer.py          # Indexação vetorial e busca
├── tools/                    # Utilitários de teste e avaliação
│   ├── scripts/              # Scripts para extração e processamento de descrições
│   ├── search_tests_results/  # Resultados de avaliação de qualidade de busca
│   └── test_results/         # Relatórios de teste de embedding
└── requirements.txt          # Dependências Python
```

## Decisões Técnicas

### 1. Escolha de Modelos e Tecnologias

#### GPT-4V para Processamento de Imagens
- **Motivo da escolha**: Capacidade superior de interpretar e descrever detalhes visuais em peças de moda
- **Benefícios**: Descrições ricas que capturam nuances de estilo, cores, padrões e elementos de design
- **Otimizações**: 
  - Implementação de prompts específicos para moda gerando descrições estruturadas em formato JSON
  - Extração de atributos categorizados (tipo de peça, cores, ocasião, estilo, etc.) para enriquecimento de metadados
  - Descrições completas em linguagem natural para melhor matching semântico

#### ChromaDB para Armazenamento Vetorial
- **Motivo da escolha**: Solução leve, eficiente e cloud-native para embeddings
- **Benefícios**: API intuitiva, excelente desempenho para bases de médio porte, persistência integrada
- **Otimizações**: 
  - Configuração de índices otimizados para busca por similaridade de coseno
  - Armazenamento de metadados ricos junto com os vetores para facilitar a filtragem e apresentação
  - Sistema de pesos personalizados para diferentes atributos durante a recuperação

#### FastAPI para Interface de API
- **Motivo da escolha**: Framework moderno, de alto desempenho e tipado
- **Benefícios**: Documentação automática, validação de dados, async/await nativo
- **Otimizações**: 
  - Implementação de endpoints assíncronos para melhor escalabilidade
  - Middleware para cache de requisições frequentes
  - Throttling inteligente para evitar sobrecarga da API OpenAI

### 2. Inovações Implementadas

#### Sistema de Pesos Dinâmicos para Atributos
- Implementação de um sistema que atribui pesos diferentes a cada atributo das peças (cor, tipo, estilo)
- Ajuste dinâmico de pesos com base na consulta do usuário (ex: consultas que mencionam cores têm peso maior no atributo cor)
- Resultados experimentais mostram melhoria de ~8% na relevância dos resultados comparado com busca sem pesos

#### Parsing Robusto de Descrições JSON
- Implementação de um parser que lida com diferentes formatações de JSON em strings markdown
- Extração inteligente de atributos com diferentes padrões de nomenclatura (camelCase, snake_case, etc.)
- Tratamento adequado para valores em diferentes formatos (strings, arrays, objetos aninhados)

#### Pipeline de Extração e Transformação de Dados
- Scripts otimizados para extração eficiente de descrições de arquivos JSON em cache
- Transformação automática em datasets estruturados para treinamento e avaliação
- Geração automática de consultas de teste baseadas nos dados extraídos

### 3. Otimizações de Performance

#### Pipeline de Normalização de Imagem
- Pré-processamento para garantir qualidade consistente (resolução, contraste, brilho)
- Uso de cache para evitar processamento repetitivo de imagens
- Implementação de processamento em lote para chamadas à API OpenAI

#### Estratégias de Caching
- Cache em memória para embeddings frequentemente acessados
- Cache de disco para descrições de imagens geradas
- Persistência de resultados de processamento intermediários com gestão de TTL (Time-To-Live)
- Otimização de I/O para leitura e escrita eficiente de arquivos de cache

#### Paralelização
- Implementação de processamento paralelo para indexação em lote
- Uso de workers assíncronos para tarefas intensivas de CPU
- Estratégia de backoff exponencial para lidar com limites de taxa da API OpenAI
- Processamento distribuído de imagens para escalabilidade horizontal

### 4. Avaliação de Qualidade

#### Métricas de Similaridade
- Desenvolvimento de sistema de avaliação de qualidade de busca
- Monitoramento de similaridade média, mínima e máxima
- Análise de distribuição de similaridade por posição no ranking
- Métricas personalizadas para avaliar relevância semântica dos resultados

#### Testes Automáticos
- Testes unitários para componentes críticos
- Testes de integração para pipeline completo
- Testes de carga para avaliar limites de escala
- Avaliação automática da qualidade das descrições geradas

## Métricas de Desempenho do Sistema

### Similaridade por Posição de Ranking
| Rank | Média | Mínimo | Máximo |
|------|-------|--------|--------|
| 1    | 0.7138| 0.6616 | 0.7608 |
| 2    | 0.7088| 0.6514 | 0.7552 |
| 3    | 0.7040| 0.6496 | 0.7525 |

### Similaridade Média por Tipo de Consulta
| Tipo de Consulta | Similaridade Média |
|------------------|-------------------|
| Consultas de Peças Específicas | 0.7294 |
| Consultas de Estilo | 0.7248 |
| Consultas de Ocasião | 0.6982 |
| Consultas de Cores | 0.7214 |
| Consultas Combinadas | 0.7056 |

## Exemplos de Consultas e Resultados

### Consulta: "Macaquinho azul escuro estilo sexy para festa de verão"
| Rank | Imagem | Tipo de Peça | Cores | Similaridade |
|------|--------|--------------|-------|--------------|
| 1 | img_238 | macaquinho | azul e branco | 0.7350 |
| 2 | img_366 | Macaquinho | Branco, verde, laranja | 0.7269 |
| 3 | img_252 | macaquinho | azul marinho e branco | 0.7264 |

### Consulta: "Vestido preto e branco elegante para ocasiões formais na primavera"
| Rank | Imagem | Tipo de Peça | Cores | Similaridade |
|------|--------|--------------|-------|--------------|
| 1 | img_205 | vestido | bege, preto | 0.6876 |
| 2 | img_471 | vestido | branco | 0.6871 |
| 3 | img_157 | vestido | branco | 0.6778 |

### Consulta: "Conjunto de top e shorts marrom e preto descontraído para uso casual no verão"
| Rank | Imagem | Tipo de Peça | Cores | Similaridade |
|------|--------|--------------|-------|--------------|
| 1 | img_390 | Conjunto de top e shorts | Vermelho e branco | 0.7069 |
| 2 | img_432 | Conjunto de duas peças (top e shorts) | Branco com estampas florais em azul, roxo, verde e rosa | 0.7063 |
| 3 | img_319 | Conjunto de top cropped e shorts | Branco, vermelho, rosa e azul | 0.7034 |

### Consulta: "Macacão curto azul claro sexy para eventos casuais de primavera/verão"
| Rank | Imagem | Tipo de Peça | Cores | Similaridade |
|------|--------|--------------|-------|--------------|
| 1 | img_238 | macaquinho | azul e branco | 0.7563 |
| 2 | img_355 | macaquinho | azul celeste | 0.7552 |
| 3 | img_3 | macaquinho | azul claro | 0.7525 |

## Instalação e Uso Rápido

### Pré-requisitos
- Python 3.8+
- Chave de API OpenAI com acesso aos modelos GPT-4V e embedding
- 8GB+ RAM recomendado para processamento de imagem

### Instalação Rápida
```bash
# Clonar o repositório
git clone <url-do-repositório>
cd teste-tecnico-fcamara

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite o arquivo .env com sua chave de API OpenAI
```

### Executando o Sistema
```bash
# Iniciar a API
python main.py --api

# Para indexar novas imagens
python main.py --index --images-dir /caminho/para/imagens

# Para executar busca via linha de comando
python main.py --search "vestido branco para primavera" --limit 5

# Para extrair descrições e gerar consultas de teste
cd tools/scripts
python extract_descriptions.py --limit 20
python generate_queries.py
```

## Visualização Completa de Resultados

Para uma visualização completa e detalhada dos resultados de teste, incluindo gráficos e métricas adicionais, consulte o relatório HTML em `tools/test_results/embedding_test_report.html`

## Próximos Passos e Melhorias Futuras

- Implementação de filtros avançados por atributos específicos (preço, tamanho, etc.)
- Refinamento do sistema de pesos com base em feedback de usuários
- Expansão do dataset com mais categorias de produtos
- Otimização do processo de geração de descrições para reduzir custos de API
- Implementação de um módulo de recomendação baseado no histórico de busca