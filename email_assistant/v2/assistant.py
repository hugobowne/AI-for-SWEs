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


# note that we expect the email template fields to match the LinkedInProfile fields
EMAIL_TEMPLATE = """\
Greetings {name},
                                 
I'm XYZ from company ABC. We're currently looking for a machine learning engineer. \
Given your experience as a {latest_role_title} and your demonstrated skills in {top_skills}, \
we thought you might be a good fit for our team!
                                  
If you're curious, feel free to pick a time in my calendar to chat. I'm eager to learn \
more about your career and what you're looking for!
"""


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file_path: str) -> State:
    """Extract text from a PDF and return it as a string."""
    reader = PdfReader(pdf_file_path)
    text = " ".join([page.extract_text() for page in reader.pages])
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["metadata", "email"])
def generate_email(state: State, response_model: type[BaseModel], response_template: str) -> State:
    text = state["pdf_text"]

    client = instructor.from_openai(openai.OpenAI())
    response = client.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the relevant metadata from the content of this PDF file."},
            {"role": "user", "content": text},
        ],
        response_model=response_model,
    )

    # `response` is an instance of `response_model`
    email = response_template.format(
        name=response.name,
        latest_role_title=response.latest_role_title,
        top_skills="; ".join(response.top_skills),
    )

    return state.update(metadata=response, email=email)


def build_assistant(app_id: str) -> Application:
    return (
        ApplicationBuilder()
        .with_actions(
            process_pdf,
            # We bind the response model and email template instead of hardcoding them
            generate_email.bind(
                response_model=LinkedInProfile,
                response_template=EMAIL_TEMPLATE,
            ),
        )
        .with_transitions(("process_pdf", "generate_email"))
        .with_entrypoint("process_pdf")
        .with_identifiers(app_id=app_id)
        .with_tracker(project="email-assistant-v2", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    app = build_assistant(app_id="test-app")
    app.visualize("assistant_v2.png", include_state=True)
