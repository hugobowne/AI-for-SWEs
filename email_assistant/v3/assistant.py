import textwrap
import openai
import instructor

from burr.core import action, ApplicationBuilder, Application, State
from pydantic import BaseModel, Field
from pypdf import PdfReader


class LinkedInProfile(BaseModel):
    """Extract metadata from a PDF export of a LinkedIn profile."""
    name: str = Field(description="First name")
    latest_role_title: str = Field(description="Job title of the most recent experience")
    top_skills: list[str] = Field(description="Top job-related skills")
    achievements: list[str] = Field(
        description="Elements from the person's work experience that demonstrate skills and constitute significant achievements"
    )


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    reader = PdfReader(pdf_file_path)
    text = " ".join([page.extract_text() for page in reader.pages])
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["metadata"])
def extract_metadata(
    state: State,
    response_model: type[BaseModel],
    instructor_client,
) -> State:
    text = state["pdf_text"]

    response = instructor_client.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the relevant metadata from the content of this PDF file."},
            {"role": "user", "content": text},
        ],
        response_model=response_model,
    )

    return state.update(metadata=response)


@action(reads=["metadata", "email"], writes=["email"])
def generate_email(state: State, instructions: str, llm_client) -> State:
    """Review the generated email and return it."""
    email = "" if state["email"] is None else state["email"]

    system_prompt = textwrap.dedent(
        """\
        You are an office assistant responsible for professional communication and drafting emails.
        Use the provided metadata and previous emails drafts to generate a new email that follows
        the user's instructions.
        """
    )

    # this multiline expression results in a concatenated string (notice that it's not a tuple with `,`)
    # here, we naively dump the metadata as a JSON string, which LLMs are decent at parsing
    # we could build a string from the metadata for better readability
    prompt = (
        f"PREVIOUS EMAIL\n{email}\n\n"
        f"METADATA\n{state['metadata'].model_dump_json()}\n\n"
        f"INSTRUCTIONS\n{instructions}"
    )

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return state.update(email=response.choices[0].message.content)


def build_assistant(app_id: str) -> Application:
    # instrumentation needs to happen before Instructor patches the LLM client
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    llm_client = openai.OpenAI()
    instructor_client = instructor.from_openai(llm_client)

    return (
        ApplicationBuilder()
        .with_actions(
            process_pdf,
            extract_metadata.bind(
                response_model=LinkedInProfile,
                instructor_client=instructor_client,
            ),
            generate_email.bind(llm_client=llm_client),
        )
        .with_transitions(
            ("process_pdf", "extract_metadata"),
            ("extract_metadata", "generate_email"),
            ("generate_email", "generate_email"),
        )
        .with_entrypoint("process_pdf")
        .with_state(State({"email": None}))
        .with_identifiers(app_id=app_id)
        .with_tracker(project="email-assistant-v3", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    app = build_assistant(app_id="test-app")
    app.visualize("assistant_v3.png", include_state=True)
