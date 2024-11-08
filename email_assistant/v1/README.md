# Email Assistant v1

Notes:
- `assistant.py` contains the Burr application
- `app.py` contains the Gradio UI
- The assistant is a simple pipeline. It has no graph cycles, no conditions, doesn't have side effects, etc.
 - Introduce Burr & features via the `ApplicationBuilder`: add tracking, add OpenTelemetry;
- Gradio doesn't hold any state or complex logic. On each execution, it creates a new Burr `Application` instance, which would map to a new `app_id` in the Burr UI. Hitting `Submit` multiple times will require parsing the PDF each time.

Exercises:
- get an LLM to do things with the content
- can you get it to output structured values?
- what happens when you ask for more or for multiple things? E.g. a haiku and extracting information.

Discussion:
- what would you test / evaluate?