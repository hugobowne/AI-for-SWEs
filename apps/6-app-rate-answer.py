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
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS feedback
                                 (id TEXT PRIMARY KEY, message_id TEXT, feedback INTEGER, 
                                  timestamp TEXT, FOREIGN KEY(message_id) REFERENCES messages(id))''')
        local.db_conn.commit()
    return local.db_conn

# Call this function on app launch to ensure the database is created upfront
get_db_connection()

# Function to start a new conversation
def start_conversation():
    conn = get_db_connection()
    c = conn.cursor()
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO conversations VALUES (?, ?)", (conversation_id, timestamp))
    conn.commit()
    print(f"New conversation started with ID: {conversation_id}")
    return conversation_id

# Function to log a message in a conversation
def log_message(conversation_id, role, content):
    conn = get_db_connection()
    c = conn.cursor()
    message_id = str(uuid.uuid4())  # Generate a new message ID
    timestamp = datetime.now().isoformat()  # Get the current timestamp
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", 
              (message_id, conversation_id, timestamp, role, content))
    conn.commit()
    return message_id  # Return the message ID properly

# Function to log feedback (thumbs-up or thumbs-down)
def log_feedback(message_id, feedback_value):
    conn = get_db_connection()
    c = conn.cursor()
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", 
              (feedback_id, message_id, feedback_value, timestamp))
    conn.commit()
    print(f"Feedback logged: {feedback_id} | Message ID: {message_id} | Feedback: {feedback_value}")

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
    # Specify a Hugging Face model for local embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
    return index

# Complete query_pdf function with proper logging of messages
def query_pdf(pdf, query, history, conversation_id, model_choice, message_id_state):
    if pdf is None:
        return [("Please upload a PDF.", "")], history, conversation_id, message_id_state
    if not query.strip():
        return [("Please enter a valid query.", "")], history, conversation_id, message_id_state

    # Start a new conversation if there isn't one
    if conversation_id is None:
        conversation_id = start_conversation()
        print(f"New conversation started with ID: {conversation_id}")

    try:
        # Choose between local (Ollama) or OpenAI model
        if model_choice == "Local (Ollama)":
            llm = Ollama(model="llama2", request_timeout=60.0)
        elif model_choice == "OpenAI":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        # Process the PDF and create an index
        index = process_pdf(pdf)
        query_engine = index.as_query_engine(llm=llm)

        # Construct the conversation string
        conversation = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
        conversation += f"\nUser: {query}\n"

        # Query the index
        response = query_engine.query(conversation)

        # Log messages and update state
        user_message_id = log_message(conversation_id, "user", query)
        assistant_message_id = log_message(conversation_id, "assistant", response.response)

        # Update conversation history
        history.append((query, response.response))
        return history, history, conversation_id, assistant_message_id
    except Exception as e:
        error_message = str(e)
        log_message(conversation_id, "system", f"Error: {error_message}")
        return [("An error occurred", error_message)], history, conversation_id, message_id_state

# Function to handle thumbs-up feedback
def handle_thumbs_up(message_id):
    if message_id:
        log_feedback(message_id, 1)  # Log thumbs-up as 1
    return "Feedback logged: 👍"

# Function to handle thumbs-down feedback
def handle_thumbs_down(message_id):
    if message_id:
        log_feedback(message_id, 0)  # Log thumbs-down as 0
    return "Feedback logged: 👎"

# Gradio interface setup
with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    model_choice = gr.Radio(label="Select Model", choices=["Local (Ollama)", "OpenAI"], value="Local (Ollama)")
    output = gr.Chatbot(label="Conversation/Query Output")
    history_state = gr.State([])  # Store conversation history
    conversation_id_state = gr.State(None)  # Store conversation ID
    message_id_state = gr.State(None)  # Store message ID for feedback

    query_button = gr.Button("Submit")

    # Feedback message output
    feedback_message = gr.Textbox(label="Feedback Status", interactive=False)

    # Feedback buttons
    with gr.Row():
        thumbs_up_button = gr.Button("👍")
        thumbs_down_button = gr.Button("👎")

    # Connect query button to query_pdf function
    query_button.click(fn=query_pdf, 
                       inputs=[pdf_upload, query_input, history_state, conversation_id_state, model_choice, message_id_state], 
                       outputs=[output, history_state, conversation_id_state, message_id_state])

    # Connect feedback buttons to logging functions
    thumbs_up_button.click(fn=handle_thumbs_up, inputs=[message_id_state], outputs=feedback_message)
    thumbs_down_button.click(fn=handle_thumbs_down, inputs=[message_id_state], outputs=feedback_message)

app.launch()