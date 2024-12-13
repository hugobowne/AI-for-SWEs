import logic


def test_extract_profile_data():
    """Tests single call to extract_profile_data."""
    linkedin_text = "John Doe\nSoftware Engineer\nSan Francisco, CA\n"
    expected = {
        "Name": "John Doe",
        "Current Role": "Software Engineer",
        "Location": "San Francisco, CA",
        "Previous Roles": "",
        "Education": "",
    }
    assert logic.extract_profile_data(linkedin_text) == expected
    # but really need to assert / evaluate individual fields
    # show pitfalls of single assert failing the whole test

# show @pytest.mark.parametrize to cover multiple cases easily

# show pytest-harvest set up to capture outputs and then to create a dataframe

# iterate