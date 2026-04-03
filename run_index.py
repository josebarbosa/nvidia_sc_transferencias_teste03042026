# -------------------------------------------------
# file: run_index.py
# -------------------------------------------------
from src.legislao_index import build_vector_index 
from pathlib import Path

# Configuração de caminhos
normas_dir    = Path("data/normas")          
index_path    = Path("faiss_index.bin")        
metadata_path = Path("metadata.json")        

# Garante que o diretório de dados existe para o teste não falhar
normas_dir.mkdir(parents=True, exist_ok=True)

# Chamar a função
try:
    build_vector_index(
        normas_dir   = normas_dir,
        index_path   = index_path,
        metadata_path= metadata_path
    )
except Exception as e:
    print(f"❌ Erro na execução: {e}")