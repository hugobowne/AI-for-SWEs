"""
To run:
pytest test_pipeline_gold.py
"""
from burr.core import State

from . import pipeline_gold

def test_process_pdf():
    state = State()
    resulting_state = pipeline_gold.process_pdf(state, "test.pdf")
    assert resulting_state.get_all() == {
        "pdf_text": "This is a test pdf.\n"
    }

def test_extract_user_data():
    pdf_text ="""
Stefan Krawczyk
CEO @ DAGWorks Inc. | Co-creator of Hamilton & Burr | Pipelines: Data, Data Science, Machine Learning, & LLMs
San Francisco, California, United States
Summary
With over 10 years of experience in building and leading data & ML related systems and teams, I am the co-founder and CEO of DAGWorks, a startup that helps data practitioners with a powerful and flexible framework based on Hamilton, an open source project that I co-created. I am also a Y Combinator alum, StartX alum, and a Stanford graduate with a Master of Science in Computer Science with Distinction in Research.
My passion is to make others more successful with data, by bridging the gap between data science, machine learning, engineering,
and business. I have expertise in data platforms, model lifecycles, featurization, and model serving systems, as well as in natural language processing, speech recognition, and spoken dialog systems. I am a proficient Python developer and a polymath engineer who can work with multiple languages and technologies.
I also contribute to open source projects and advise other startups
in the data space. I value good leadership, company culture, and innovation, and I strive to be a humble and effective leader who empowers and enables my team to solve challenging and interesting problems.
Experience
DAGWorks Inc.
Chief Executive Officer
October 2022 - Present (2 years 1 month) San Francisco Bay Area
Standardizing how people write python for data, machine learning, and LLM pipelines/applications. Built on https://github.com/dagworks-inc/hamilton & https://github.com/dagworks-inc/burr.
    """
    state = State(dict(pdf_text=pdf_text))
    resulting_state = pipeline_gold.extract_user_data(state)
    li_profile = resulting_state["user_data"]
    assert li_profile.name == "Stefan Krawczyk"
    assert li_profile.latest_role_title == "CEO"
    # this next one is tricky -- the order of the skills can vary; the number too
    # what you do depends on your requirements
    assert len(set(li_profile.top_skills).intersection({"Data", "Data Science", "Machine Learning"})) == 3

# Advanced:
# use pytest fixtures
# requires pytest-harvest plugin
def test_my_agent(results_bag):
    results_bag.input = "my_value"
    results_bag.output = "my_output"
    results_bag.expected_output = "my_expected_output"

def test_print_results(module_results_df):
    # place this function at the end of the module so that way it's run last.
    print(module_results_df.columns) # this will include "input", "output", "expected_output"
    print(module_results_df.head()) # this will show the first few rows of the results
    # TODO: Add more evaluation logic here or log the results to a file, etc.
    # assert some threshold of success, etc.
    module_results_df.to_csv("./test_results.csv")