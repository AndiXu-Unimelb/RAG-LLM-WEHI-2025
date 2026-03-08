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
    return "Search stopped.", "Stopped"



import time
import threading

def ask(q):
    global stop_flag
    stop_flag = False

    if not q.strip():
        yield "Please enter a question.", "0.0 s"
        return

    result_container = {}

    def run_search():
        docs = retriever.invoke(q)

        if not docs:
            result_container["result"] = "No results found."
            return

        result = qa_chain.invoke(q)

        answer = result["answer"]
        sources = result["sources"]

        if sources:
            sources_text = "\n".join(f"- {s}" for s in sources)
        else:
            sources_text = "No sources found."

        result_container["result"] = f"""Answer:
{answer}

Sources:
{sources_text}
"""

    thread = threading.Thread(target=run_search)
    thread.start()

    start_time = time.time()

    while thread.is_alive():
        if stop_flag:
            yield "Search stopped.", "Stopped"
            return

        elapsed = time.time() - start_time
        yield "Searching...", f"{elapsed:.1f} s"
        time.sleep(0.1)

    thread.join()

    total_time = time.time() - start_time
    yield result_container.get("result", "Unknown error."), f"{total_time:.2f} s"


with gr.Blocks() as ui:
    gr.Markdown("## Student Organizer RAG LLM")

    input_box = gr.Textbox(label="Question")
    output_box = gr.Textbox(label="Answer", lines=15)

    with gr.Row():
        ask_btn = gr.Button("Search")
        stop_btn = gr.Button("Stop")

    timer_box = gr.Markdown("⏱ 0.0 s")

    ask_btn.click(
        ask,
        inputs=input_box,
        outputs=[output_box, timer_box]
    )

    stop_btn.click(
        stop_search,
        outputs=[output_box, timer_box]
    )

ui.launch()
