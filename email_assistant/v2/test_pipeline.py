"""
To run:
pytest test_pipeline.py
"""
from burr.core import State

from . import pipeline

def test_process_pdf():
    state = State()
    resulting_state = pipeline.process_pdf(state, "test.pdf")
    assert resulting_state.get_all() == {
        "pdf_text": "This is a test pdf.\n"
    }

def test_extract_user_data():
    # TODO:
    pdf_text ="""
    """
    state = State(dict(pdf_text=pdf_text))
    resulting_state = pipeline.extract_user_data(state)
    li_profile = resulting_state["user_data"]
    assert li_profile.name == "TODO"
