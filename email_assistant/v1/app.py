import pathlib
import re
import uuid

import gradio as gr
from assistant import build_assistant


def _create_app_id(file_path: str) -> str:
    """Create a session ID based on the file path."""
    safe_file_name = re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(file_path).name)
    return f"{safe_file_name[:20]}-{str(uuid.uuid4())[:8]}"


def run_assistant(pdf_file: str, query_input: str) -> str | None:
    """Run the assistant on the latest user message and current PDF file.
    This doesn't leverage the full conversation history.
    """
    if pdf_file:
        assistant = build_assistant(app_id=_create_app_id(pdf_file))
        action_name, results, state = assistant.run(
            halt_after=["generate_email"],
            inputs={"pdf_file_path": pdf_file, "instructions": query_input}
        )
        reply = state["llm_reply"]
    else:
        gr.Info("Please upload a PDF file to continue.")
        reply = None

    return reply


def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="PDF source", type="filepath", file_types=[".pdf"])
                query_input = gr.Textbox(label="Instructions", max_lines=100)
                query_button = gr.Button("Submit", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="LLM", max_lines=100)
        query_button.click(run_assistant, [pdf_input, query_input], text_output)

    return ui


if __name__ == "__main__":
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    ui = build_ui()
    ui.launch(server_port=8111)
