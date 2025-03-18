#!/usr/bin/env python3
# Script para corrigir caminhos em arquivos que usam caminhos relativos
import os
import re
import sys

def fix_file(file_path, replacements):
    with open(file_path, 'r') as f:
        content = f.read()
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Corrigido: {file_path}")

# Corrigir generate_test_queries.py
generate_queries_path = "/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/tools/scripts/generate_test_queries.py"
fix_file(generate_queries_path, [
    (r'\.\.(/|\\)data(/|\\)input(/|\\)test_descriptions\.csv', r'/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/src/data/input/test_descriptions.csv'),
    (r'\.\.(/|\\)data(/|\\)input(/|\\)test_queries\.txt', r'/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/src/data/input/test_queries.txt')
])

# Corrigir test_embeddings.py
test_embeddings_path = "/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/tools/tests/test_embeddings.py"
fix_file(test_embeddings_path, [
    (r'\.\.(/|\\)data(/|\\)input(/|\\)test_queries\.txt', r'/home/azureuser/fashion-similarity-search/teste-tecnico-fcamara/src/data/input/test_queries.txt'),
    # Adicionar parâmetro para reutilizar embeddings
    (r'def main\(\):', r'''def main():
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Reutiliza embeddings existentes sem recalcular")'''),
    (r'run_embedding_tests\(queries, args.top_k, args.output_dir\)', r'run_embedding_tests(queries, args.top_k, args.output_dir, reuse_embeddings=args.reuse_embeddings)')
])

print("Caminhos corrigidos com sucesso!")
