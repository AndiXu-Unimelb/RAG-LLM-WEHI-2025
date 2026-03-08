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



stop_flag = False


def stop_search():
    global stop_flag
    stop_flag = True
    return "Search stopped."



def ask(q):
    global stop_flag
    stop_flag = False

    if not q.strip():
        yield "Please enter a question."
        return

    yield "Searching..."

    docs = retriever.invoke(q)

    if stop_flag:
        yield "Search stopped."
        return

    if not docs:
        yield "No results found."
        return

    result = qa_chain.invoke(q)

    if stop_flag:
        yield "Search stopped."
        return

    answer = result["answer"]
    sources = result["sources"]

    if sources:
        sources_text = "\n".join(f"- {s}" for s in sources)
    else:
        sources_text = "No sources found."

    yield f"""Answer:
{answer}

Sources:
{sources_text}
"""


with gr.Blocks() as ui:
    gr.Markdown("## Student Organizer RAG LLM")

    input_box = gr.Textbox(label="Question")
    output_box = gr.Textbox(label="Answer", lines=15)

    with gr.Row():
        ask_btn = gr.Button("Search")
        stop_btn = gr.Button("Stop")

    ask_btn.click(ask, inputs=input_box, outputs=output_box)
    stop_btn.click(stop_search, outputs=output_box)

ui.launch()
