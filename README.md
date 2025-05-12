This repo is a WIP. We'll teach our first iteration of this workshop in Nov 2024 at the [MLOps World and Generative AI World Conference](https://generative-ai-summit.com/).

## Description:
This workshop is designed to equip software engineers with the skills to build and iterate on generative AI-powered applications. Participants will explore key components of the AI software development lifecycle through first principles thinking, including prompt engineering, monitoring, evaluations, and handling non-determinism. The session focuses on using multimodal AI models to build applications, such as querying PDFs, while providing insights into the engineering challenges unique to AI systems. By the end of the workshop, participants will know how to build a PDF-querying app, but all techniques learned will be generalizable for building a variety of generative AI applications.

If you're a data scientist, machine learning practitioner, or AI enthusiast, this workshop can also be valuable for learning about the software engineering aspects of AI applications, such as lifecycle management, iterative development, and monitoring, which are critical for production-level AI systems.

## What You'll Learn:
- How to integrate AI models and APIs into a practical application.
- Techniques to manage non-determinism and optimize outputs through prompt engineering.
- How to monitor, log, and evaluate AI systems to ensure reliability.
- The importance of handling structured outputs and using function calling in AI models.
- The software engineering side of building AI systems, including iterative development, debugging, and performance monitoring.
- Practical experience in building an app to query PDFs using multimodal models.


## Workshop Prerequisite Knowledge:
- Basic programming knowledge in Python.
- Familiarity with REST APIs.
- Experience working with Jupyter Notebooks or similar environments (preferred but not required).
- No prior experience with AI or machine learning is required.
- Most importantly, a sense of curiosity and a desire to learn!

If you have a background in data science, ML, or AI, this workshop will help you understand the software engineering side of building AI applications.

## 🛠️ Setup Instructions (GitHub Codespaces)

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### 1. Install Dependencies

Poetry is pre-installed in Codespaces. From the root of the repo, run:

```bash
poetry install
```

### 2. Activate the Environment

To use the virtual environment for normal `python` commands:

```bash
source $(poetry env info --path)/bin/activate
```

After that, you can run scripts like:

```bash
python apps/1-app-query.py
```
