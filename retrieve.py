import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_PATH,
        embedding_function=embedding_model
    )

    print("Vectorstore loaded successfully.")
    return vectorstore

def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )
    return retriever

def retrieve_chunks(query, retriever):
    chunks = retriever.invoke(query)
    return chunks

# Test
if __name__ == "__main__":
    vs = load_vectorstore()
    retriever = get_retriever(vs)
    results = retrieve_chunks("ποιά είναι τα μαθήματα του πρώτου εξαμήνου", retriever)
    
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")