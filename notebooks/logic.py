import json

import openai

client = openai.Client()


def extract_profile_data(linkedin_text: str) -> dict:
    messages = [
        {"role": "system",
         "content": "You are an expert in extracting structured information from text."},
        {"role": "user",
         "content": f"""Extract the following structured information from the text below:
- Name
- Current Role
- Location
- Previous Roles
- Education
Text: 
{linkedin_text}
Output the result as a JSON object.
"""}
    ]

    # Make LLM API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=messages
    )
    raw_output = response.choices[0].message.content
    return json.loads(raw_output)