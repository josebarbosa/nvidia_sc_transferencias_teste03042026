import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1) Configuração de caminhos
index_path = "faiss_index.bin"
meta_path  = "metadata.json"

# 2) Carrega o índice e metadados (CORRIGIDO)
if not Path(index_path).exists() or not Path(meta_path).exists():
    print("❌ Erro: Arquivos de índice ou metadata não encontrados. Rode o indexador primeiro.")
    exit()

# No FAISS, usamos read_index diretamente do módulo
index = faiss.read_index(str(Path(index_path)))

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

# -------------------------------------------------
# Função de busca (sem depender de REST)
# -------------------------------------------------
def retrieve(query: str, k: int = 3):
    # Modelo de embedding (deve ser o mesmo usado na indexação)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Gera o vetor da query
    query_vec = model.encode([query], normalize_embeddings=True).astype('float32')
    
    # Busca no índice (D = distâncias/similaridades, I = índices dos vizinhos)
    D, I = index.search(query_vec, k)
    
    # Extrai os resultados baseados nos índices retornados
    results_text = [meta[i]["text"] for i in I[0] if i != -1]
    sources = [meta[i]["source"] for i in I[0] if i != -1]

    return sources, results_text

# -----------------------------------------
# Exemplo de teste
# -----------------------------------------
query = "qual o modelo de licitação mais usado para valores entre 80 e 150 mil reais?"

# Chamada corrigida para o nome da função definida acima
sources, excerpts = retrieve(query)

print("\n" + "="*40)
print("=== Busca por trechos relevantes ===")
print("="*40)

if not sources:
    print("Nenhum resultado encontrado.")
else:
    for i in range(len(sources)):
        print(f"▶ Fonte ({i+1}): {sources[i]}")
        # Limita a exibição do trecho para não poluir o terminal
        content = excerpts[i].replace('\n', ' ')
        print(f"   Trecho: {content[:200]}...")
        print("-" * 20)