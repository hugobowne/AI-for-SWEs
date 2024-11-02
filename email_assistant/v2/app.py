import gradio as gr
from assistant import build_assistant, _generate_model


def add_user_message(history: list[dict], message: dict, pdf_file: str) -> tuple:
    """Add the user message to conversation history, display it, and extract PDF file paths."""
    pdf_file_path = pdf_file

    for x in message["files"]:
        # display the file as the user's message
        history.append({"role": "user", "content": {"path": x}})
        # add the file to the list of PDFs if is a PDF
        if x.endswith(".pdf"):
            pdf_file_path = x

    # display the user's message
    if message["text"]:
        history.append({"role": "user", "content": message["text"]})

    # the order of returned components must match the order of `chat_input.submit()`
    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.Files(value=pdf_file_path),
    )


def run_assistant(history: list[dict], pdf_file: str, response_format: dict) -> list[dict]:
    """Run the assistant on the latest user message and current PDF file.
    This doesn't leverage the full conversation history.
    """
    instructions = history[-1]["content"]  # retrieve the last user message
    # TODO handle when last message was a file; `content` will be a dict instead of str

    if pdf_file:
        assistant = build_assistant()

        response_format_model = _generate_model(response_format)

        action_name, results, state = assistant.run(
            halt_after=["generate_email"],
            inputs={
                "pdf_file_path": pdf_file,
                "instructions": instructions,
                "response_format": response_format_model
            }
        )

        print(state["llm_reply"])
        history.append({"role": "assistant", "content": gr.JSON(state["llm_reply"].model_dump_json())})
    else:
        history.append({"role": "assistant", "content": "Please upload a PDF file to continue."})

    return history


def _update_fields(fields: dict, name: str, description: str, type_: str) -> tuple:
    fields[name] = {"name": name, "description": description, "type": type_}
    return fields, fields


def response_format_form() -> gr.State:
    with gr.Blocks():
        fields = gr.State({})
        with gr.Row():  
            with gr.Column():
                json_output = gr.JSON(value={}, label="Fields to extract")

            with gr.Column():
                field_name = gr.Textbox(label="Name")
                field_description = gr.Textbox(label="Description")
                field_type = gr.Dropdown(choices=["str", "int", "float", "bool"], label="Type")

                add_field_btn = gr.Button("Add field")
                add_field_btn.click(
                    _update_fields,
                    [fields, field_name, field_description, field_type],
                    [fields, json_output]
                ).then(
                    lambda: ("", ""), None, [field_name, field_description]
                )

                @gr.render(inputs=[fields])
                def delete_field(field_values: dict):
                    if field_values:
                        field_selection = gr.Dropdown(
                            label="Field to delete",
                            choices=list(field_values.keys()),
                            interactive=True
                        )
                        delete_btn = gr.Button("Delete field")
                        delete_btn.click(
                            lambda f: f.pop(field_selection), fields, fields
                        )
    return fields


def build_ui() -> gr.Blocks:
    """Return a Gradio UI for the email assistant."""
    with gr.Blocks() as ui:
        chatbot = gr.Chatbot(label="Email Assistant", type="messages", height=800, elem_id="chatbot")
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="single",
            placeholder="Send a message or upload a file",
            show_label=False,
        )

        
        with gr.Tab("Response format"):
            response_format = response_format_form()

        with gr.Tab("Files"):
            pdf_files = gr.File(value=None, label="Uploaded PDF", interactive=False)

        # add the user message to history and display it
        # then run the assistant
        # then refresh the chat input
        chat_input.submit(
            add_user_message, [chatbot, chat_input, pdf_files], [chatbot, chat_input, pdf_files]
        ).then(
            run_assistant, [chatbot, pdf_files, response_format], chatbot
        ).then(
            lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
        )

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_port=8111)
