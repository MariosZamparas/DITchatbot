import config
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path):
    pdf_loader = PyPDFDirectoryLoader(docs_path)
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} pages from '{docs_path}'")
    return documents


def chunk_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

def embed_and_store(chunks):
    
    print(f"Embedding {len(chunks)} chunks...")

    embedding_model = HuggingFaceEmbeddings(
    model_name= config.EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=config.VECTORSTORE_PATH
    )

    print(f"Stored {len(chunks)} chunks in vectorstore at '{config.VECTORSTORE_PATH}'")    
    return vectorstore

def run_ingestion():
    if os.path.exists(config.VECTORSTORE_PATH):
        print("Vectorstore already exists. Delete it to re-ingest.")
        return
     
    docs = load_documents(config.PDF_PATH)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)


__main__ = run_ingestion()
