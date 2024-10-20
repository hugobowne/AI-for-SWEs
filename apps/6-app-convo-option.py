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

# Thread-local storage for database connections
local = threading.local()

# Function to get a thread-local database connection
def get_db_connection():
    if not hasattr(local, "db_conn"):
        local.db_conn = sqlite3.connect('qa_traces.db', check_same_thread=False)
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS conversations
                                 (id TEXT PRIMARY KEY, timestamp TEXT)''')
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS messages
                                 (id TEXT PRIMARY KEY, conversation_id TEXT, 
                                  timestamp TEXT, role TEXT, content TEXT,
                                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
        local.db_conn.commit()
    return local.db_conn

# Function to log a message in a conversation
def log_message(conversation_id, role, content):
    conn = get_db_connection()
    c = conn.cursor()
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", 
              (message_id, conversation_id, timestamp, role, content))
    conn.commit()

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file_bytes):
    pdf_doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to process the uploaded PDF and create an index
def process_pdf(pdf_file_bytes):
    extracted_text = extract_text_from_pdf(pdf_file_bytes)
    document = Document(text=extracted_text)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
    return index

# Function to handle query based on mode (Single Query or Conversation)
def query_pdf(pdf, query, history, conversation_id, model_choice, mode_choice):
    if pdf is None:
        return [("Please upload a PDF.", "")], history, conversation_id
    if not query.strip():
        return [("Please enter a valid query.", "")], history, conversation_id
    
    if mode_choice == "Conversation" and conversation_id is None:
        conversation_id = str(uuid.uuid4())  # Start a new conversation if needed
    
    try:
        # Choose between local (Ollama) or OpenAI model
        if model_choice == "Local (Ollama)":
            llm = Ollama(model="llama2", request_timeout=60.0)
        elif model_choice == "OpenAI":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        index = process_pdf(pdf)
        query_engine = index.as_query_engine(llm=llm)

        if mode_choice == "Single Query":
            # In Single Query Mode, no conversation context is used
            response = query_engine.query(query)
            return [(query, response.response)], [], None  # Return response without history

        elif mode_choice == "Conversation":
            # In Conversation Mode, maintain conversation history
            conversation = ""
            for h in history:
                conversation += f"User: {h[0]}\nAssistant: {h[1]}\n"
            conversation += f"User: {query}\n"
            
            response = query_engine.query(conversation)
            log_message(conversation_id, "user", query)
            log_message(conversation_id, "assistant", response.response)
            
            history.append((query, response.response))
            return history, history, conversation_id  # Return with updated history and conversation ID
    
    except Exception as e:
        error_message = str(e)
        return [("An error occurred", error_message)], history, conversation_id

# Gradio interface setup
with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    model_choice = gr.Radio(label="Select Model", choices=["Local (Ollama)", "OpenAI"], value="Local (Ollama)")
    mode_choice = gr.Radio(label="Mode", choices=["Single Query", "Conversation"], value="Single Query")
    output = gr.Chatbot(label="Conversation/Query Output")
    history_state = gr.State([])  # Store conversation history
    conversation_id_state = gr.State(None)  # Store conversation ID
    
    query_button = gr.Button("Submit")
    query_button.click(fn=query_pdf, 
                       inputs=[pdf_upload, query_input, history_state, conversation_id_state, model_choice, mode_choice], 
                       outputs=[output, history_state, conversation_id_state])

app.launch()