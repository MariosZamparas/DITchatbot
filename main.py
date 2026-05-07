import os
from langchain_core.messages import HumanMessage, AIMessage
from retrieve import load_vectorstore, get_retriever
from prompt import build_prompt_template
from agent import load_llm, build_chain, ask

def main():
    # Guard — make sure ingestion has been run first
    if not os.path.exists("vectorstore/"):
        print("Vectorstore not found. Please run ingest.py first.")
        return

    print("Loading components...")

    # Setup — runs once at startup
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    prompt_template = build_prompt_template()
    llm = load_llm()
    chain = build_chain(prompt_template, llm)

    print("Ready! Type your question. Type '0' to exit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input == "0":
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = ask(user_input, chain, retriever, llm, history)

            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response))

            print(f"\nAssistant: {response}\n")

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()