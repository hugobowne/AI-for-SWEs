# Email Assistant v2

Notes:
- We introduce structured output and how to add guardrails around LLM's response
- Introduce Burr features via the `ApplicationBuilder`: add tracking, add OpenTelemetry; use `.bind()` on action
- Show Burr UI and how we can view the structured output models
- Gradio has a more complex dynamic form to create a structured output model
- Gradio doesn't hold any state or complex logic. On each execution, it creates a new Burr `Application` instance, which would map to a new `app_id` in the Burr UI. Hitting `Submit` multiple times will require parsing the PDF each time.