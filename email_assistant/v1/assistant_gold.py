import openai
from burr.core import action, ApplicationBuilder, Application, State
import pymupdf


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    pdf_doc = pymupdf.open(filename=pdf_file_path, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["llm_reply"])
def generate_email(state: State, instructions: str) -> State:
    """Generate answer based on the PDF's text using an LLM following the instructions"""
    text = state["pdf_text"]
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ],
    )

    return state.update(llm_reply=response.choices[0].message.content)


def build_assistant(app_id: str) -> Application:
    return (
        ApplicationBuilder()
        .with_actions(process_pdf, generate_email)
        .with_transitions(("process_pdf", "generate_email"))
        .with_entrypoint("process_pdf")
        .with_identifiers(app_id=app_id)
        .with_tracker(project="email-assistant-v1", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    app = build_assistant(app_id="test-app")
    app.visualize("assistant_v1.png", include_state=True)
