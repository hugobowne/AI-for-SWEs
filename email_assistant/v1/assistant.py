import openai
from burr.core import action, ApplicationBuilder, Application, State
from pypdf import PdfReader


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    reader = PdfReader(pdf_file_path)
    text = " ".join([page.extract_text() for page in reader.pages])
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


def build_assistant() -> Application:
    return (
        ApplicationBuilder()
        .with_actions(process_pdf, generate_email)
        .with_transitions(("process_pdf", "generate_email"))
        .with_entrypoint("process_pdf")
        .build()
    )


if __name__ == "__main__":
    app = build_assistant()
    app.visualize("assistant_v1.png", include_state=True)
