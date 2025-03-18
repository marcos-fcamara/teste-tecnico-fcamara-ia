# Sistema de Busca por Similaridade de Imagens de Moda com IA

Um sistema nativo em nuvem que permite busca em linguagem natural para itens de moda, aproveitando os modelos de visão e embeddings da OpenAI para entender e combinar descrições de roupas. O sistema processa imagens de moda, gera descrições detalhadas e habilita busca semântica através de embeddings vetoriais.

## Decisões Técnicas e Ferramentas Utilizadas

### Arquitetura Geral
- **Arquitetura modular** separando processamento de imagens, armazenamento e API
- **Abordagem em pipeline** com normalização > descrição > embedding > indexação
- **Banco de dados vetorial** para buscas semânticas eficientes

### Componentes Principais

1. **Processamento de Imagens**
  - **OpenAI GPT-4 Vision**: Escolhido pela capacidade superior de descrição detalhada de imagens
  - **Normalização de Imagens**: Pipeline personalizado para melhorar imagens de baixa resolução (220x220)
  - **Técnicas de cache**: Reduz chamadas repetidas à API, economizando custos

2. **Embeddings e Busca Semântica**
  - **OpenAI text-embedding-3-large**: Embeddings de alta dimensionalidade para maior precisão
  - **ChromaDB com FAISS**: Banco vetorial com algoritmo HNSW para buscas rápidas
  - **Parâmetros otimizados**: construction_ef=128, search_ef=96, M=16 para melhor qualidade/velocidade

3. **API e Interface**
  - **FastAPI**: Framework web de alta performance com validação automática
  - **Processamento em lotes**: Otimizado para 5 imagens por lote
  - **Paralelização**: Uso de ThreadPoolExecutor para maior throughput

4. **Ferramentas de Teste**
  - **Sistema de avaliação**: Testes automatizados de qualidade dos embeddings
  - **Geração de consultas**: Uso da API do GPT para criar consultas realistas de teste

## Exemplos de Consultas e Resultados

### Consulta: "Vestido floral para verão"

| Imagem | Similaridade | Tipo de Peça | Cores | Descrição |
|--------|--------------|--------------|-------|-----------|
| ![vestido1](src/data/images/Hot-selling-fashion-lace-halter-neck-Backless-print-cross-spaghetti-strap-one-piece-dress-short-skirt.jpg_220x220.jpg) | 0.89 | Vestido | Azul e branco | Vestido com estampa floral, tecido leve, ideal para verão |
| ![vestido2](src/data/images/Skine-2015-new-casual-fashion-summer-dress-ball-gown-white-and-black-printed-sleeveless-spaghetti-strap.jpg_220x220.jpg) | 0.85 | Vestido | Azul, branco, preto | Vestido casual com estampa floral, alças finas, estilo verão |
| ![vestido3](src/data/images/2015-summer-new-Women-s-Cool-sexy-dress-beach-floral-print-short-sleeveless-mini-dress.jpg_220x220.jpg) | 0.82 | Macaquinho | Verde, rosa, amarelo | Peça com estampa floral e folhagens tropicais para clima quente |

### Consulta: "Macaquinho preto elegante"

| Imagem | Similaridade | Tipo de Peça | Cores | Descrição |
|--------|--------------|--------------|-------|-----------|
| ![macaquinho1](src/data/images/2015-Hot-New-Arrival-Spring-Womens-Sleeveless-Short-Jumpsuit-Printed-Leopard-Strapless-Bodycon-Dress-Tank-Tops.jpg_220x220.jpg) | 0.78 | Macaquinho | Marrom, preto | Macaquinho com estampa de animal, sensual e ousado |
| ![macaquinho2](src/data/images/new-top-2015-hot-fashion-deep-v-neck-sexy-women-rompers-sexty-casual-women-jumpsuits-solid.jpg_220x220.jpg) | 0.75 | Macaquinho | Preto | Macaquinho com decote profundo, tecido leve, elegante |
| ![macaquinho3](src/data/images/Free-Shipping-New-Fashion-Hot-Style-Deep-V-Neck-Half-Sleeves-Lace-Short-Rompers-Women-Jumpsuit.jpg_220x220.jpg) | 0.72 | Macaquinho | Azul marinho, branco | Macaquinho com detalhes em renda, prático e elegante |

### Consulta: "Roupa estilo boho para primavera"

| Imagem | Similaridade | Tipo de Peça | Cores | Descrição |
|--------|--------------|--------------|-------|-----------|
| ![boho1](src/data/images/Vestidos-Women-Summer-Lace-Dress-2014-Sheer-Floral-Lace-Maxi-Slip-with-Wide-Straps-Black-Romance.jpg_220x220.jpg) | 0.81 | Vestido | Azul, bege | Vestido estilo boho, romântico, com detalhes em renda |
| ![boho2](src/data/images/Women-Knitted-Long-Sleeve-Lace-Dresses-Spring-And-winter-Sexy-vestidos-Chiffon-Patchwork-Dress-Casual-Ladies.jpg_220x220.jpg) | 0.77 | Vestido | Bege e preto | Vestido com renda, estilo boho, para primavera |
| ![boho3](src/data/images/macacao-feminino-2015-Colorful-print-rompers-womens-jumpsuit-Casual-overalls-for-women-bodysuit-shorts-for-women.jpg_220x220.jpg) | 0.74 | Conjunto | Rosa, azul, branco | Conjunto tie-dye descontraído e veranil, estilo boho |