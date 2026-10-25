import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
CHUNK_SIZE = 500  # en tokens
CHUNK_OVERLAP = 50  # en tokens

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./vectorstore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee todos los .md y .txt de la carpeta y los convierte en Documents."""
    documents = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Divide los documentos en chunks de CHUNK_SIZE tokens con CHUNK_OVERLAP de solapamiento."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_embeddings() -> OllamaEmbeddings:
    """Modelo de embeddings compartido por la ingesta y la consulta."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_vectorstore() -> Chroma:
    """Abre (o crea) la colección persistente de ChromaDB."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def ingest(reset: bool = False) -> Chroma:
    """Indexa /data en ChromaDB, salvo que la colección ya tenga datos."""
    vectorstore = get_vectorstore()
    already_indexed = len(vectorstore.get(limit=1)["ids"]) > 0

    if already_indexed and not reset:
        print(f"La colección '{COLLECTION_NAME}' ya existe en {CHROMA_DIR}. No se re-indexa.")
        return vectorstore

    if already_indexed and reset:
        print("Borrando la colección para re-indexar...")
        vectorstore.reset_collection()

    chunks = split_documents(load_documents())
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]

    print(f"Generando embeddings con '{EMBEDDING_MODEL}' para {len(chunks)} chunks...")
    vectorstore.add_documents(chunks, ids=ids)
    print(f"Listo: {len(chunks)} chunks guardados en {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest(reset="--reset" in sys.argv)
