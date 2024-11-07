# import textwrap
import openai
# import instructor

from burr.core import action, ApplicationBuilder, Application, State
from pydantic import BaseModel, Field
import pymupdf


class LinkedInProfile(BaseModel):
    """Extract metadata from a PDF export of a LinkedIn profile."""
    name: str = Field(description="First name")
    latest_role_title: str = Field(description="Job title of the most recent experience")
    top_skills: list[str] = Field(description="Top job-related skills")
    achievements: list[str] = Field(
        description="Elements from the person's work experience that demonstrate skills and constitute significant achievements"
    )
    # add more here, e.g.
    # specializations: list[str] = Field(description="Field specializations that are clear from their work history. E.g. 'Machine Learning', 'Data Science', etc.")


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
    response_model: type[BaseModel],
    structured_llm_client,
) -> State:
    text = state["pdf_text"]

    response = structured_llm_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the relevant user data from the content of this PDF file."},
            {"role": "user", "content": text},
        ],
        response_format=response_model,
    )
    parsed_data = response.choices[0].message.parsed
    return state.update(user_data=parsed_data)


@action(reads=["user_data"], writes=["email", "chat_history"])
def generate_email(state: State, system_prompt: str, instructions: str, llm_client) -> State:
    """Review the generated email and return it."""

    # this multiline expression results in a concatenated string (notice that it's not a tuple with `,`)
    # here, we naively dump the user_data as a JSON string, which LLMs are decent at parsing
    # we could build a string from the user_data for better readability
    prompt = (
        f"USER DATA\n{state['user_data'].model_dump_json()}\n\n"
        f"INSTRUCTIONS\n{instructions}"
    )
    chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
    )
    chat_history.append(
        {"role": response.choices[0].message.role,
         "content": response.choices[0].message.content}
    )
    return state.update(email=response.choices[0].message.content, chat_history=chat_history)

@action(reads=["chat_history"], writes=["chat_history"])
def user_feedback(state: State, feedback: str) -> State:
    chat_history = state["chat_history"]
    chat_history.append(
        {"role": "user", "content": feedback}
    )
    return state.update(chat_history=chat_history)


@action(reads=["user_data", "email", "chat_history"], writes=["email", "chat_history"])
def iterate_on_email(state: State, llm_client) -> State:
    """Review the generated email and return it."""
    chat_history = state["chat_history"]
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
    )
    email = response.choices[0].message.content
    chat_history.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })
    return state.update(email=email, chat_history=chat_history)


def build_assistant(app_id: str) -> Application:
    # instrumentation needs to happen before Instructor patches the LLM client
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()

    llm_client = openai.OpenAI() # swap this for ollama if needed
    # instructor_client = instructor.from_openai(llm_client)

    return (
        ApplicationBuilder()
        .with_actions(
            process_pdf,
            extract_user_data.bind(
                response_model=LinkedInProfile,
                structured_llm_client=llm_client,
            ),
            generate_email.bind(llm_client=llm_client),
            user_feedback,
            iterate_on_email.bind(llm_client=llm_client)

        )
        .with_transitions(
            ("process_pdf", "extract_user_data"),
            ("extract_user_data", "generate_email"),
            ("generate_email", "user_feedback"),
            ("user_feedback", "iterate_on_email"),
            ("iterate_on_email", "user_feedback"),
        )
        .with_entrypoint("process_pdf")
        .with_state(State({"email": None, "chat_history": []}))
        .with_identifiers(app_id=app_id)
        .with_tracker(project="email-assistant-v3", use_otel_tracing=True)
        .build()
    )


if __name__ == "__main__":
    app = build_assistant(app_id="test-app")
    app.visualize("assistant_v3.png", include_state=True)
