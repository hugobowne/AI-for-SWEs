import pathlib
import re
import uuid

import gradio as gr
from assistant import build_assistant


def _create_app_id(file_path: str) -> str:
    """Create a session ID based on the file path."""
    safe_file_name = re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(file_path).name)
    return f"{safe_file_name[:20]}-{str(uuid.uuid4())[:8]}"


def run_assistant(pdf_file: str) -> tuple[str | None, dict]:
    """Run the assistant on the latest user message and current PDF file.
    This doesn't leverage the full conversation history.
    """
    if pdf_file:
        assistant = build_assistant(app_id=_create_app_id(pdf_file))
        action_name, results, state = assistant.run(
            halt_after=["generate_email"],
            inputs={"pdf_file_path": pdf_file}
        )
        reply = state["email"]
        user_data = state["user_data"].model_dump_json()
    else:
        gr.Info("Please upload a PDF file to continue.")
        reply = None
        user_data = {}

    return reply, user_data


def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""
    with gr.Blocks() as ui:
        pdf_input = gr.File(label="PDF source", type="filepath", file_types=[".pdf"])  # Handle file
        query_button = gr.Button("Generate", variant="primary")
        
        with gr.Row():
            with gr.Column():
                text_output = gr.Textbox(label="Email", max_lines=100, show_copy_button=True)
            with gr.Column():
                user_data_output = gr.JSON(label="Extracted user_data", height=400)

        query_button.click(run_assistant, [pdf_input], [text_output, user_data_output])

    return ui


if __name__ == "__main__":
    import os
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    ui = build_ui()
    ui.launch(server_port=8111)
