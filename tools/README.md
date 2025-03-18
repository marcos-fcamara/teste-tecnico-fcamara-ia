# Fashion Image Search Testing Suite - A Comprehensive Testing Framework for Fashion E-commerce Search Systems

This project provides a robust testing framework for evaluating and validating fashion image search systems. It combines text-based queries, image embeddings, and similarity testing to ensure accurate and relevant search results in fashion e-commerce applications.

The suite includes tools for generating diverse test queries using AI, processing image descriptions, testing embedding quality, and evaluating search accuracy. It supports both general fashion searches and specialized queries based on categories like style, occasion, color, and season. The framework generates detailed reports with visualizations and metrics to help analyze and improve search system performance.

## Repository Structure
```
teste-tecnico-fcamara/
└── tools/
    ├── data/input/                 # Input data files for testing
    │   ├── ai_generated_queries.json   # AI-generated test queries
    │   ├── test_descriptions.html      # HTML formatted product descriptions
    │   ├── test_descriptions.txt       # Text formatted product descriptions
    │   └── test_queries.txt           # Plain text test queries
    ├── results/                    # Test results and reports
    │   ├── embeddings/             # Embedding test results and visualizations
    │   ├── search/                 # Search test results
    │   └── similarity/             # Similarity test results
    ├── scripts/                    # Core testing utilities
    │   ├── extract_descriptions.py     # Extracts product descriptions
    │   ├── generate_test_queries.py    # Generates AI-powered test queries
    │   └── simple_queries.py           # Basic query generation
    └── tests/                      # Test implementation
        ├── test_embeddings.py         # Image embedding tests
        └── test_search.py             # Search functionality tests
```

## Usage Instructions
### Prerequisites
- Python 3.8 or higher
- OpenAI API key for query generation
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - tabulate
  - openai
  - tqdm
  - python-dotenv

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd teste-tecnico-fcamara

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install pandas numpy matplotlib tabulate openai tqdm python-dotenv

# Set up environment variables
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Quick Start
1. Generate test queries:
```bash
cd tools/scripts
python generate_test_queries.py
```

2. Run embedding tests:
```bash
cd tools/tests
python test_embeddings.py
```

3. View results:
```bash
# Open the embedding test report
open ../results/embeddings/embedding_test_report.html
```

### More Detailed Examples
1. Generate specialized queries for specific categories:
```python
from scripts.generate_test_queries import QueryGenerator

generator = QueryGenerator()
df = pd.read_csv("data/input/test_descriptions.csv")

# Generate occasion-specific queries
occasion_queries = generator.generate_specialized_queries(
    df, 
    num_queries=5, 
    category_type="ocasião"
)
```

2. Run comprehensive search tests:
```python
from tests.test_search import run_search_tests

results = run_search_tests(
    queries_file="data/input/test_queries.txt",
    top_k=3
)
```

### Troubleshooting
1. OpenAI API Issues
- Error: "API key not found"
  - Check if .env file exists and contains OPENAI_API_KEY
  - Verify environment variable is loaded: `echo $OPENAI_API_KEY`

2. Test Data Loading Issues
- Error: "File not found"
  - Ensure you're in the correct directory
  - Verify file paths in scripts match repository structure

3. Embedding Test Failures
- Check if vector database is populated
- Verify image processor initialization
- Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Data Flow
The framework processes fashion product data through a pipeline of description extraction, query generation, and search testing.

```ascii
[Product Descriptions] -> [Description Extraction] -> [Query Generation]
           |                                                |
           v                                                v
    [Image Processing] <- [Search Testing] <- [Test Execution]
           |                                                |
           v                                                v
    [Vector Database] -> [Similarity Testing] -> [Test Reports]
```

Key component interactions:
1. Description Extractor processes JSON cache files to create structured product data
2. Query Generator uses OpenAI API to create diverse test queries
3. Image Processor handles embedding generation and vector storage
4. Search Testing executes queries against the vector database
5. Results Processor generates detailed HTML reports with visualizations
6. Test Framework coordinates all components and manages data flow
7. Report Generator creates visualizations and metrics for analysis