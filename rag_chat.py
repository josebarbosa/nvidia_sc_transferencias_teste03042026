import json
import torch
import faiss
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    GenerationConfig,
    BitsAndBytesConfig
)

# -------------------------------------------------
# 1. Configurações e Carregamento de Dados
# -------------------------------------------------
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.json"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 

def load_system():
    print("\n" + "="*60)
    print("🚀 INICIALIZANDO SISTEMA RAG (GPU 4-BIT) 🚀")
    print("="*60)

    print("--- Carregando Índice e Metadata ---")
    index = faiss.read_index(str(Path(INDEX_PATH)))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("--- Carregando Modelos (RTX 5070 Mode) ---")
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )

    gen_config = GenerationConfig.from_pretrained(LLM_MODEL)
    gen_config.max_new_tokens = 512
    gen_config.temperature = 0.1
    gen_config.do_sample = True
    gen_config.pad_token_id = tokenizer.eos_token_id

    # Pipeline de geração
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return index, metadata, embedder, generator, gen_config

# -------------------------------------------------
# 2. Lógica de Busca e Geração
# -------------------------------------------------

def get_answer(query, index, metadata, embedder, generator, gen_config):
    # Recuperação
    query_vec = embedder.encode([query], normalize_embeddings=True).astype('float32')
    _, I = index.search(query_vec, k=3)
    
    contexts = []
    sources = []
    for i in I[0]:
        if i != -1:
            contexts.append(metadata[i]["text"])
            sources.append(metadata[i]["source"])
    
    context_str = "\n\n".join(contexts)
    unique_sources = list(set(sources))

    # Prompt
    prompt = f"""<s>[INST] VOCÊ É UM AUDITOR ESPECIALISTA EM LEGISLAÇÃO DE SANTA CATARINA.
RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.

Use o CONTEXTO abaixo para responder à pergunta de forma técnica e objetiva.

CONTEXTO:
{context_str}

PERGUNTA:
{query} [/INST] Resposta em Português:</s>"""

    output = generator(prompt, generation_config=gen_config)
    answer = output[0]['generated_text'].split("[/INST]")[-1].replace("Resposta em Português:</s>", "").strip()
    
    return answer, unique_sources

# -------------------------------------------------
# 3. Loop de Chat no Console
# -------------------------------------------------

def main():
    try:
        index, metadata, embedder, generator, gen_config = load_system()
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
        return

    print("\n✅ Sistema Pronto! Digite sua pergunta ou 'sair' para encerrar.")
    
    while True:
        print("\n" + "-"*30)
        query = input("❓ Pergunta: ").strip()

        if query.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando chat. Até logo!")
            break
        
        if not query:
            continue

        print("🔍 Analisando decretos e gerando resposta...")
        
        try:
            resposta, fontes = get_answer(query, index, metadata, embedder, generator, gen_config)
            
            print(f"\n💡 RESPOSTA:\n{resposta}")
            print(f"\n📂 FONTES: {', '.join(fontes)}")
        except Exception as e:
            print(f"❌ Ocorreu um erro durante a geração: {e}")

if __name__ == "__main__":
    main()