import openai
from burr.core import action, ApplicationBuilder, Application, State
from burr.integrations.opentelemetry import init_instruments
from pydantic import BaseModel, create_model, Field
from pypdf import PdfReader


def _generate_model(fields: dict[str, dict]) -> type[BaseModel]:
    field_mapping = {
        name: (info["type"], Field(description=info["description"]))
        for name, info in fields.items()
    }
    return create_model("ResponseFormat", **field_mapping)


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    reader = PdfReader(pdf_file_path)
    text = " ".join([page.extract_text() for page in reader.pages])
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["llm_reply"])
def generate_email(
    state: State,
    instructions: str,
    response_format: type[BaseModel],
    client: openai.OpenAI
) -> State:
    """Generate answer based on the PDF's text using an LLM following the instructions"""
    text = state["pdf_text"]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ],
        response_format=response_format,
    )
    
    message = response.choices[0].message
    if message.parsed:
        print("ACCEPTED")
        new_state = state.update(llm_reply=message.parsed)
    else:
        print("REFUSED")
        new_state = state.update(llm_reply=message.refusal)

    return new_state


def build_assistant() -> Application:
    client = openai.OpenAI()
    init_instruments("openai")

    return (
        ApplicationBuilder()
        .with_actions(
            process_pdf,
            generate_email.bind(client=client)
        )
        .with_transitions(("process_pdf", "generate_email"))
        .with_entrypoint("process_pdf")
        .with_tracker(project="email-assistant-v2", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    app = build_assistant()
    app.visualize("assistant_v2.png", include_state=True)
