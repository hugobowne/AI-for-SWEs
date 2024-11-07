import pathlib
import re
import textwrap
import uuid
from os import system

import gradio as gr
from burr.core import Application
from fsspec.registry import default

from assistant import build_assistant


def _create_app_id(file_path: str) -> str:
    """Create a session ID based on the file path."""
    safe_file_name = re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(file_path).name)
    return f"{safe_file_name[:20]}-{str(uuid.uuid4())[:8]}"


def run_assistant(assistant: Application,
                  pdf_file: str,
                  system_prompt: str | None,
                  instructions: str | None) -> tuple[str | None, dict]:
    instructions = "" if instructions is None else instructions
    system_prompt = "" if system_prompt is None else system_prompt
    if pdf_file:
        assistant = build_assistant(app_id=_create_app_id(pdf_file))
        action_name, results, state = assistant.run(
            halt_after=["generate_email"],
            inputs={"pdf_file_path": pdf_file,
                    "instructions": instructions,
                    "system_prompt": system_prompt}
        )
        # reply = state["email"]
        initial_history = state["chat_history"]
        metadata = state["user_data"].model_dump_json()
    else:
        gr.Info("Please upload a PDF file to continue.")
        # reply = None
        metadata = {}
        initial_history = None

    return initial_history, metadata, assistant

def iterate_on_email(assistant: Application, feedback: str) -> tuple[str | None, dict]:
    action_name, results, state = assistant.run(
        halt_before=["user_feedback"],
        inputs={"feedback": feedback}
    )
    # reply = state["email"]
    return state["chat_history"]

default_system_prompt = textwrap.dedent(
        """\
        You are an office assistant responsible for professional communication and drafting emails.
        Use the provided user data and any prior context to drafts to generate a new email that follows
        the user's instructions.
        """
    )

def reset_state(assistant, pdf_input, system_prompt, initial_instructions, text_output, metadata_output, refine_box):
    assistant = gr.State(None)
    pdf_input = None
    system_prompt = default_system_prompt
    initial_instructions = None
    text_output= None
    metadata_output = None
    refine_box = None
    return assistant, pdf_input, system_prompt, initial_instructions, text_output, metadata_output, refine_box


def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""


    with gr.Blocks() as ui:
        # reset_button = gr.Button("Reset app", variant="secondary")
        assistant = gr.State(None)

        with gr.Group():
            pdf_input = gr.File(label="PDF source", type="filepath", file_types=[".pdf"])
            system_prompt = gr.Textbox(label="System prompt", lines=2, show_copy_button=True,
                                       value=default_system_prompt)
            initial_instructions = gr.Textbox(label="Instructions", lines=2, show_copy_button=True)
            query_button = gr.Button("Generate initial email", variant="primary")
        
        with gr.Row():
            with gr.Column():
                # text_output = gr.Textbox(label="Email", max_lines=100, show_copy_button=True)
                text_output = gr.Chatbot(label="Conversation", type="messages")
                refine_box = gr.Textbox(label="Refine Email", lines=2, show_copy_button=True)
                refine_button = gr.Button("Iterate on email", variant="primary")
            with gr.Column():
                metadata_output = gr.JSON(label="Extracted user data", height=400)


        pdf_input.upload(
            lambda pdf_file_path: build_assistant(_create_app_id(pdf_file_path)),
            inputs=pdf_input,
            outputs=assistant
        )

        query_button.click(
            run_assistant,
            inputs=[assistant, pdf_input, system_prompt, initial_instructions],
            outputs=[text_output, metadata_output, assistant]
        )

        refine_button.click(
            iterate_on_email,
            [assistant, refine_box],
            outputs=[text_output]
        )

        # reset_button.click(
        #     reset_state,
        #     [assistant, pdf_input, system_prompt, initial_instructions, text_output, metadata_output, refine_box],
        #     [assistant, pdf_input, system_prompt, initial_instructions, text_output, metadata_output, refine_box]
        # )
        
        text_output.change(lambda x: gr.update(visible=x is not None), initial_instructions, initial_instructions)

    return ui


if __name__ == "__main__":
    import os
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    ui = build_ui()
    ui.launch(server_port=8111)
