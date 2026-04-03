# -------------------------------------------------
# file: src/legislao_index.py
# -------------------------------------------------
import json
from pathlib import Path
from typing import List, Dict

# Bibliotecas que fazem o trabalho de extração e embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------
# 2️⃣  Função principal que cria o índice vetorial
# -------------------------------------------------
def build_vector_index(
        normas_dir: Path,
        index_path: Path,
        metadata_path: Path,
        *,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 3000,
        overlap: int = 200) -> None:
    """
    - Lê todos os arquivos *.txt* dentro de `normas_dir`.
    - Divide cada documento em chunks.
    - Gera embeddings com sentence-transformers.
    - Cria um índice vetorial FAISS.
    """

    # -------------------------------------------------
    # 1️⃣  Ler todos os arquivos *.txt* dentro de `normas_dir`
    # -------------------------------------------------
    txt_files = list(normas_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nenhum *.txt encontrado em {normas_dir}")

    # ---------- 2️⃣ Fragmentar cada documento ----------
    # Usando o caminho de importação atualizado
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks: List[Dict[str, str]] = []   

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            raw = f.read()
            for i, chunk in enumerate(splitter.split_text(raw)):
                chunks.append({
                    "source": str(txt_file),
                    "chunk_id": i,
                    "text": chunk
                })

    # -------------------------------------------------
    # 3️⃣ Gerar embeddings para cada chunk
    # -------------------------------------------------
    # Extraímos apenas o texto para o modelo de embedding
    texts = [c["text"] for c in chunks]
    
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    # ---------- 4️⃣ Construir o índice vetorial FAISS ----------
    d = embeddings.shape[1]                 # dimensão do vetor
    index = faiss.IndexFlatIP(d)            # similaridade por Produto Interno (Cosseno se normalizado)
    index.add(np.array(embeddings).astype('float32')) 

    # -------------------------------------------------
    # 5️⃣ Salvar índice e metadata
    # -------------------------------------------------
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # Correção do método de escrita do FAISS
    faiss.write_index(index, str(index_path))

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as out:
        json.dump(chunks, out, ensure_ascii=False, indent=2)

    print(f"✅ Índice criado e salvo em {index_path}")
    print(f"🗂️  {len(chunks)} chunks indexados")
    print(f"📄 Metadata salvo em {metadata_path}")