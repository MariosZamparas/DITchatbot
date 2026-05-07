from langchain_core.prompts import ChatPromptTemplate

def build_prompt_template():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         
        Εισαι βοηθος που απανταει σε ερωτησεις σχετικα με την σχολη Πληροφορικης και τηλεπικοινωνιων του Πανεπιστημιου Αρτας. 
        Απαντα μονο σε ερωτησεις σχετικα με την σχολη και μην απαντας σε κανενα αλλο θεμα. 
        Αν δεν γνωριζεις την απαντηση σε μια ερωτηση, απαντα "Δεν γνωριζω".

        Context: {context}
         
"""),
        ("human", "{input}"),
    ])
    return prompt

def format_context(chunks):
    formatted = ""
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        formatted += f"[{i+1}] (Source: {source})\n{chunk.page_content}\n\n"
    return formatted