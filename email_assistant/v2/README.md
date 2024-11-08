# Email Assistant v2

Notes:
- We introduce structured output and how to add guardrails around LLM's response
- Show Burr UI and how we can view the structured output models
- Gradio has a more complex dynamic form to create a structured output model
- Gradio doesn't hold any state or complex logic. On each execution, it creates a new Burr `Application` instance, which would map to a new `app_id` in the Burr UI. Hitting `Submit` multiple times will require parsing the PDF each time.

Exercises:
 - Complete "pipeline"
 - Play with the structured output model - add / remove
   - does adding more make it perform well?
   - what happens if something isn't in the context?
 - Write unit tests & test fixtures
 - What should be a hard failure?

Discussion:
 - why are the structured outputs useful?
 - When does it make sense for hard failures vs soft? How does this impact your SDLC?

