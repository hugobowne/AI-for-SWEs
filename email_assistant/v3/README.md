# Email Assistant v3

Notes:
- `gr.update()` allows to change the visibility of UI components; useful to change the inputs based on the current action
- With the new Burr graph structure, we want to call `.run()` and halt multiple time for the application and `app_id`. This allows us to have a human in the loop and add feedback. Then, we need to add logic in Gradio to determine when to change the `app_id`.
- we define a default state because of the loop over `generate_email` where we read/write to the `email` field
- the app is rebuilt if a new PDF is uploaded