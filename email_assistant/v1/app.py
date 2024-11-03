import gradio as gr
from assistant import build_assistant


def run_pipeline(pdf_file: str, session_id: gr.State, query_input: str) -> list[dict]:
    """Run the pipeline on the latest user message and current PDF file.
    This doesn't leverage the full conversation history.
    """
    if pdf_file:
        assistant = build_assistant(session_id.value)
        action_name, results, state = assistant.run(
            halt_after=["generate_response"],
            inputs={"pdf_file_path": pdf_file, "instructions": query_input}
        )

        history = [{"role": "assistant", "content": state["llm_reply"]}]
    else:
        history = [{"role": "assistant", "content": "Please upload a PDF file to continue."}]

    return history

def create_session_id(pdf_upload) -> gr.State:
    """Create a session ID for the assistant using the uploaded file name stripping any characters
    that would interfere with a file path."""
    file_name = pdf_upload.name.split("/")[-1]
    name = file_name.replace(" ", "_").replace(".", "_")
    # trim to UUID length - going from the end
    if len(name) > 36:
        name = name[len(name) - 36:]
    return gr.State(name)

def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""
    with gr.Blocks() as ui:
        pdf_upload = gr.File(label="Upload PDF", type="filepath")  # Handle file
        query_input = gr.Textbox(label="Type prompt instructions here")
        response_box = gr.Chatbot(label="Response", type="messages", height=400, elem_id="response_box")
        session_id_value = gr.State(None)  # Store conversation ID
        query_button = gr.Button("Submit")
        # create the session id based on the uploaded file name
        # then run the pipeline and update the response box
        query_button.click(
            create_session_id, [pdf_upload], [session_id_value]
        ).then(
            run_pipeline, [pdf_upload, session_id_value, query_input], response_box
        )

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_port=8111)
