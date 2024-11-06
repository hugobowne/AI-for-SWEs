import pathlib
import re
import uuid

import gradio as gr
from burr.core import Application

from assistant import build_assistant


def _create_app_id(file_path: str) -> str:
    """Create a session ID based on the file path."""
    safe_file_name = re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(file_path).name)
    return f"{safe_file_name[:20]}-{str(uuid.uuid4())[:8]}"


def run_assistant(assistant: Application, pdf_file: str, instructions: str | None) -> tuple[str | None, dict]:
    instructions = "" if instructions is None else instructions
    if pdf_file:    
        action_name, results, state = assistant.run(
            halt_after=["generate_email"],
            inputs={"pdf_file_path": pdf_file, "instructions": instructions}
        )
        reply = state["email"]
        metadata = state["metadata"].model_dump_json()
    else:
        gr.Info("Please upload a PDF file to continue.")
        reply = None
        metadata = {}

    return reply, metadata


def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""
    with gr.Blocks() as ui:       
        assistant = gr.State(None)

        with gr.Group():
            pdf_input = gr.File(label="PDF source", type="filepath", file_types=[".pdf"])
            feedback_input = gr.Textbox(label="Instructions", lines=2, show_copy_button=True)
            query_button = gr.Button("Generate", variant="primary")
        
        with gr.Row():
            with gr.Column():
                text_output = gr.Textbox(label="Email", max_lines=100, show_copy_button=True)
            with gr.Column():
                metadata_output = gr.JSON(label="Extracted metadata", max_height=800)
        
        pdf_input.upload(
            lambda pdf_file_path: build_assistant(_create_app_id(pdf_file_path)),
            inputs=pdf_input,
            outputs=assistant
        )

        query_button.click(
            run_assistant,
            inputs=[assistant, pdf_input, feedback_input],
            outputs=[text_output, metadata_output]
        )
        
        text_output.change(lambda x: gr.update(visible=x is not None), feedback_input, feedback_input)

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_port=8111)
