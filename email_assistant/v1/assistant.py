from burr.core import action, ApplicationBuilder, Application, State
from burr.core.graph import Graph, GraphBuilder
from burr.tracking import LocalTrackingClient


@action(reads=[], writes=["pdf_text"])
def process_pdf(state: State, pdf_file: bytes) -> State:
    document = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")
    
    return state.update(pdf_text=text)


@action(reads=["pdf_text"], writes=[])
def store_document(state: State) -> State:
    return state


@action(reads=[], writes=["llm_answer"])
def answer_question(state: State, user_query: str) -> State:
    return ...



def build_graph() -> Graph:
    return (
        GraphBuilder()
        .with_actions(
            process_pdf,
            store_document,
            answer_question
        )
        .with_transitions(
            ("process_pdf", "store_document"),
            ("store_document", "answer_question"),
        )
        .build()
    )



def build_assistant() -> Application:
    graph = build_graph()
    return (
        ApplicationBuilder()
        .with_graph(graph)
        .with_identifiers(app_id="...")
        .initialize_from(
            initializer=LocalTrackingClient(project="ai-for-swes"),
            resume_at_next_action=True,
            default_entrypoint="next-action",
            default_state={"conversation": []},
        )
        .build()
    )


if __name__ == "__main__":
    assistant_graph = build_graph()
    assistant_graph.visualize("assistant_v1.png", include_state=True)

