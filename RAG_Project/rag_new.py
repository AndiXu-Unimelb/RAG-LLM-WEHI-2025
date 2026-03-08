from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="bling-phi-3")


template = """
{context}

Question: {question}

Please answer the question using ONLY the information provided above.
If the information is insufficient, say that you do not know.
"""

prompt = PromptTemplate.from_template(template)



qa_chain = (
    RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    )
    | {
        "answer": prompt | llm | StrOutputParser(),
        "sources": lambda x: list(
            {
                doc.metadata.get("source", "Unknown")
                for doc in x["context"]
            }
        ),
    }
)



def ask(q):
    if not q.strip():
        return "Please enter a question."

    result = qa_chain.invoke(q)

    answer = result["answer"]
    sources = result["sources"]

    if sources:
        sources_text = "\n".join(f"- {s}" for s in sources)
    else:
        sources_text = "No sources found."

    return f"""Answer:
{answer}

Sources:
{sources_text}
"""

ui = gr.Interface(fn=ask, inputs="text", outputs=gr.Textbox(label="Answer",lines=15), title="Student Organizer RAG LLM")
ui.launch()
