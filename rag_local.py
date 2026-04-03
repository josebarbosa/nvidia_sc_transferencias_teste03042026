import json
import torch
import faiss
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

print("--- Carregando Índice e Metadata ---")
index = faiss.read_index(str(Path(INDEX_PATH)))
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("--- Carregando Modelos (Otimizado para RTX 5070 12GB) ---")
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Configuração de Quantização 4-bit (bitsandbytes)
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
    device_map="cuda:0" # Força o uso da GPU RTX
)

# Configurações de Geração
gen_config = GenerationConfig.from_pretrained(LLM_MODEL)
gen_config.max_new_tokens = 512
gen_config.temperature = 0.1
gen_config.do_sample = True
gen_config.pad_token_id = tokenizer.eos_token_id

# -------------------------------------------------
# 2. Funções do Sistema RAG
# -------------------------------------------------

def retrieve_context(query: str, k: int = 3):
    query_vec = embedder.encode([query], normalize_embeddings=True).astype('float32')
    _, I = index.search(query_vec, k)
    
    contexts = []
    sources = []
    for i in I[0]:
        if i != -1:
            contexts.append(metadata[i]["text"])
            sources.append(metadata[i]["source"])
    
    return "\n\n".join(contexts), list(set(sources))

def generate_answer(query: str):
    context, sources = retrieve_context(query)
    
    # Prompt com reforço de idioma e persona
    prompt = f"""<s>[INST] VOCÊ É UM AUDITOR ESPECIALISTA EM LEGISLAÇÃO DE SANTA CATARINA.
RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.

Use os trechos dos Decretos Estaduais abaixo para responder à pergunta de forma objetiva.
Se a informação não estiver no contexto, diga que não encontrou informações suficientes.

### CONTEXTO:
{context}

### PERGUNTA:
{query} [/INST] Resposta em Português:</s>"""

    print("\n--- Gerando resposta (Inferência em 4-bit na GPU) ---")
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # Execução da geração
    output = generator(prompt, generation_config=gen_config)
    
    full_text = output[0]['generated_text']
    answer = full_text.split("[/INST]")[-1].replace("Resposta em Português:</s>", "").strip()
    
    return answer, sources

# -------------------------------------------------
# 3. Execução
# -------------------------------------------------
if __name__ == "__main__":
    pergunta = "Qual o prazo para entregar a prestação de contas de um convênio?"
    
    resposta, fontes = generate_answer(pergunta)
    
    print("\n" + "="*60)
    print(f"PERGUNTA: {pergunta}")
    print("-" * 60)
    print(f"RESPOSTA:\n{resposta}")
    print("-" * 60)
    print("FONTES UTILIZADAS:")
    for f in fontes:
        print(f"- {f}")
    print("="*60)