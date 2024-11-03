from typing import Optional

import openai
from burr.core import action, ApplicationBuilder, Application, State
import pymupdf
from burr.tracking import LocalTrackingClient

# this will auto instrument the openAI client. No swapping of imports required!
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.ollama import OllamaInstrumentor

OpenAIInstrumentor().instrument()
OllamaInstrumentor().instrument()


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    pdf_doc = pymupdf.open(pdf_file_path, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["llm_reply"])
def generate_response(state: State, instructions: str) -> State:
    """Generate answer based on the PDF's text using an LLM following the instructions"""
    text = state["pdf_text"]
    client = openai.OpenAI() # replace this with ollama as needed.

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ],
    )

    return state.update(llm_reply=response.choices[0].message.content)


def build_assistant(burr_app_id: Optional[str]) -> Application:
    tracker = LocalTrackingClient(project="ai-for-swes-email-v1")
    return (
        ApplicationBuilder()
        .with_actions(process_pdf, generate_response)
        .with_transitions(("process_pdf", "generate_response"))
        .initialize_from(
            tracker,
            resume_at_next_action=False,
            default_state={},
            default_entrypoint="process_pdf",
        ).with_identifiers(app_id=burr_app_id)
        .with_tracker(tracker, use_otel_tracing=True)  # tracking + checkpointing; one line 🪄.
        .build()
    )


if __name__ == "__main__":
    app = build_assistant(None)
    app.visualize("assistant_v1.png", include_state=True)
