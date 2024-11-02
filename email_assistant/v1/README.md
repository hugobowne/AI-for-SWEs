# Email Assistant v1

Notes:
- `assistant.py` contains the Burr application
- `app.py` contains the Gradio UI
- The assistant is a simple pipeline. It has no graph cycles, no conditions, doesn't have side effects, etc.
- Show a very limited set of Burr features as an introduction (no OpenTelemetry, no tracking, no `GraphBuilder`, etc.)
- Gradio doesn't hold any state or complex logic. On each execution, it creates a new Burr `Application` instance, which would map to a new `app_id` in the Burr UI. Hitting `Submit` multiple times will require parsing the PDF each time.

Exercises:
- 