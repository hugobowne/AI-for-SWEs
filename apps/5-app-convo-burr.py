import copy

from IPython.display import Image, display
from IPython.core.display import HTML
import openai
import os
import gradio as gr
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import fitz  # PyMuPDF
import sqlite3
from datetime import datetime
import threading
import uuid

from burr.core import ApplicationBuilder, State, default, graph, when
from burr.core.action import action
from burr.tracking import LocalTrackingClient


# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file_bytes):
    pdf_doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text


@action(reads=[], writes=["vector_store"])
def process_pdf(state: State, pdf_file_bytes: bytes) -> State:
    extracted_text = extract_text_from_pdf(pdf_file_bytes)
    document = Document(text=extracted_text)
    # Specify a Hugging Face model for local embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
    return state.update(vector_store=index)


@action(reads=["vector_store", "conversation_history"], writes=["conversation_history"])
def query_vector_store(state: State, query: str, model_choice: str) -> State:
    if model_choice == "Local (Ollama)":
        llm = Ollama(model="llama2", request_timeout=60.0)
    elif model_choice == "OpenAI":
        # Use OpenAI's model, get API key from environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    else:
        raise ValueError(f"Invalid model choice {model_choice}")
    index = state["vector_store"]
    query_engine = index.as_query_engine(llm=llm)

    conversation_history = state["conversation_history"]
    conversation = ""
    # TODO: fix this up to fit how llama index needs it, or drop llama index, or drop the need
    # for adding user history.
    for h in conversation_history:
        conversation += f"User: {h[0]}\nAssistant: {h[1]}\n"
    conversation += f"User: {query}\n"

    response = query_engine.query(conversation)
    conversation_history.append([query, response.response])
    return state.update(conversation_history=conversation_history)


def build_graph():
    # Built the graph.
    base_graph = (
        graph.GraphBuilder()
        .with_actions(
            # these are the "nodes"
            process_pdf=process_pdf,
            query_vector_store=query_vector_store,
        )
        .with_transitions(
            # these are the edges between nodes, based on state.
            ("process_pdf", "query_vector_store", default),
            ("query_vector_store", "process_pdf", default),
        )
        .build()
    )
    # base_graph.visualize()
    return base_graph

def build_application(base_graph, conversation_id):
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    # this will auto instrument the openAI client. No swapping of imports required!
    OpenAIInstrumentor().instrument()
    from opentelemetry.instrumentation.ollama import OllamaInstrumentor
    OllamaInstrumentor().instrument()

    tracker = LocalTrackingClient(project="ai-for-swes")
    app = (
        ApplicationBuilder()
        .with_graph(base_graph)
        .initialize_from(
            tracker,
            resume_at_next_action=True,
            default_state={"conversation_history": []},
            default_entrypoint="process_pdf",
        )
        .with_tracker(tracker, use_otel_tracing=True)  # tracking + checkpointing; one line 🪄.
        .with_identifiers(app_id=conversation_id)
        .build()
    )
    return app

graph = build_graph()

def query_pdf(pdf, query, history, conversation_id, model_choice):
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    burr_app = build_application(graph, conversation_id)
    _, _, app_state = burr_app.run(
        halt_after=["query_vector_store"],
        inputs={"pdf_file_bytes": pdf, "query": query, "model_choice": model_choice}
    )
    # run `burr` on the commandline to spin up the UI and see the logs.
    return app_state["conversation_history"], app_state["conversation_history"], conversation_id

# Gradio interface setup
with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")  # Handle file as binary
    query_input = gr.Textbox(label="Ask a question about the PDF")
    model_choice = gr.Radio(label="Select Model", choices=["Local (Ollama)", "OpenAI"], value="Local (Ollama)")
    output = gr.Chatbot(label="Conversation")
    history_state = gr.State([])  # Store conversation history
    conversation_id_state = gr.State(None)  # Store conversation ID

    query_button = gr.Button("Submit")
    query_button.click(fn=query_pdf,
                       inputs=[pdf_upload, query_input, history_state, conversation_id_state, model_choice],
                       outputs=[output, history_state, conversation_id_state])

app.launch()