import openai
# import instructor

from burr.core import action, ApplicationBuilder, Application, State
from pydantic import BaseModel, Field
import pymupdf


class LinkedInProfile(BaseModel):
    """Extract user data from a PDF export of a LinkedIn profile."""
    name: str = Field(description="First name")
    # TODO: add more fields here to match requirements for template


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
    pdf_doc = pymupdf.open(filename=pdf_file_path, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=["user_data"])
def extract_user_data(
        state: State,
) -> State:
    text = state["pdf_text"]
    system_prompt = "Extract the relevant data from the content of this PDF file."
    # client = instructor.from_openai(openai.OpenAI())
    client = openai.OpenAI()
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format=LinkedInProfile,
    )
    li_profile: LinkedInProfile = response.choices[0].message.parsed
    return state.update(user_data=li_profile)


@action(reads=["user_data"], writes=["email"])
def generate_email(state: State) -> State:
    li_profile: LinkedInProfile = state["user_data"]
    email = EMAIL_TEMPLATE.format(
        name=li_profile.name,
        latest_role_title=li_profile.latest_role_title,
        top_skills="; ".join(li_profile.top_skills),
    )
    return state.update(user_data=li_profile, email=email)


def build_pipeline(app_id: str) -> Application:
    return (
        ApplicationBuilder()
        .with_actions(
            process_pdf,
            extract_user_data,
            generate_email,
        )
        .with_transitions(("process_pdf", "extract_user_data"),
                          ("extract_user_data", "generate_email"))
        .with_entrypoint("process_pdf")
        .with_identifiers(app_id=app_id)
        .with_tracker(project="email-assistant-v2", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    app = build_pipeline(app_id="test-app")
    app.visualize("assistant_v2.png", include_state=True)
