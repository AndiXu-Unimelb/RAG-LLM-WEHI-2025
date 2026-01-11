from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./chromadb", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="phi3:mini")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask(q):
    if not q.strip():
        return "Please enter a question."
    return qa_chain.run(q)

ui = gr.Interface(fn=ask, inputs="text", outputs="text", title="Student Organizer RAG LLM")
ui.launch()
