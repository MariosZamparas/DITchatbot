import config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from retrieve import load_vectorstore, get_retriever, retrieve_chunks
from prompt import build_prompt_template, format_context

def load_llm():
    llm = ChatOllama(model=config.OLLAMA_MODEL)
    return llm

def build_chain(prompt_template, llm):
    chain = prompt_template | llm
    return chain

def rewrite_query(query, llm):
    rewrite_prompt = f"""You are helping search a Greek university document.
    Rewrite the following question using formal Greek university terminology 
    that would appear in an official university guide. 
    Return ONLY the rewritten query, nothing else.

    Original question: {query}
    Rewritten query:"""

    response = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()
    print(f"Rewritten query: {rewritten}")
    return rewritten

def ask(query, chain, retriever, llm, history):
    rewritten_query = rewrite_query(query, llm)
    chunks = retrieve_chunks(rewritten_query, retriever)
    context = format_context(chunks)

    print("\n===== CONTEXT SENT TO LLM =====")
    print(context)
    print("================================\n")

    response = chain.invoke({
        "input": query,  # original query — the LLM answers what the user actually asked
        "context": context,
        "history": history,
    })

    return response.content