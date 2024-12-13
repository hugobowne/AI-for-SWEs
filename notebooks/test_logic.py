import pytest
import logic
import json
import os

# Define the mode at the top of the file: "strict" or "non-strict"
mode = "strict"  # Change this to "non-strict" for the original function

# Define the file path at the top of the file
path = "data/stefanLI.txt"  # Default file path (can be changed here)

# Expected values for each file
expected_values = {
    "data/stefanLI.txt": {
        "Name": "Stefan Krawczyk",
        "Current Role": "CEO @ DAGWorks Inc.",
        "Location": "San Francisco, California, United States",
        "Previous Roles": "",
        "Education": "",
    },
    "data/hbaLI.txt": {
        "Name": "Hugo Bowne-Anderson",
        "Current Role": "Independent Data and AI Scientist",
        "Location": "Darlinghurst, New South Wales, Australia",
        "Previous Roles": [
            {"Title": "Head of Developer Relations", "Company": "Outerbounds", "Duration": "Feb 2022 - Aug 2024"},
            {"Title": "Head of Data Science Evangelism and Marketing", "Company": "Coiled", "Duration": "May 2020 - Oct 2021"},
        ],
        "Education": [
            {"Degree": "Doctor of Philosophy (PhD)", "Field": "Pure Mathematics", "Institution": "UNSW", "Years": "2006 - 2011"},
            {"Degree": "Bachelor of Science (B.S.) (First Class Honors)", "Field": "Mathematics, English Literature", "Institution": "University of Sydney", "Years": "2001 - 2005"},
        ],
    },
}

# Helper function to load text from a file
def load_text_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Select the appropriate function based on the mode
def extract_profile_data_based_on_mode(linkedin_text):
    if mode == "strict":
        return logic.extract_profile_data_strict(linkedin_text)
    elif mode == "non-strict":
        return logic.extract_profile_data(linkedin_text)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'strict' or 'non-strict'.")

# Test if the output is valid JSON using a file
def test_valid_json_with_file():
    assert os.path.exists(path), f"File does not exist: {path}"
    linkedin_text = load_text_from_file(path)
    raw_output = extract_profile_data_based_on_mode(linkedin_text)

    try:
        parsed_output = json.loads(json.dumps(raw_output))
        assert isinstance(parsed_output, dict), "Output is not a valid JSON object"
    except json.JSONDecodeError:
        pytest.fail("Output is not valid JSON")

# Test if the extracted JSON contains the required fields using a file
def test_field_validation_with_file():
    assert os.path.exists(path), f"File does not exist: {path}"
    linkedin_text = load_text_from_file(path)
    assert path in expected_values, f"No expected values defined for file: {path}"

    expected = expected_values[path]
    output = extract_profile_data_based_on_mode(linkedin_text)

    # Validate individual fields
    for field, expected_value in expected.items():
        if isinstance(expected_value, list):
            assert sorted(output.get(field, [])) == sorted(expected_value), f"Field '{field}' does not match. Expected: {expected_value}, Got: {output.get(field)}"
        else:
            assert output.get(field) == expected_value, f"Field '{field}' does not match. Expected: {expected_value}, Got: {output.get(field)}"

# Test for consistency across iterations
@pytest.mark.parametrize("iterations", [5])  # Number of iterations
def test_no_variability_with_file(iterations):
    assert os.path.exists(path), f"File does not exist: {path}"
    linkedin_text = load_text_from_file(path)
    outputs = [extract_profile_data_based_on_mode(linkedin_text) for _ in range(iterations)]

    # Check for consistency
    assert len(set(json.dumps(output) for output in outputs)) == 1, "Outputs vary across iterations"