import sys
import torch

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✅ Importação de LangChain bem-sucedida!")
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ Importação de LangChain (legacy path) bem-sucedida!")
    except ImportError as e:
        print(f"❌ Erro ao importar: {e}")

# Verificação de CUDA
print(f"Versão do Python: {sys.version}")
print(f"CUDA disponível no Torch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")