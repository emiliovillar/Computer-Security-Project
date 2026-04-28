# Zero-to-Hired

## Overview

Zero-to-Hired is a CV-reviewer demo for studying prompt injection in document-processing pipelines. The app now supports three contribution areas:

- A data sanitization pipeline using `llm-guard` plus keyword-based redaction.
- Semantic output protection using NeMo Guardrails on the final evaluation step.
- Baseline multi-model comparison across three Groq-hosted models with protections disabled.

## Models

The app uses these Groq model IDs:

- `llama-3.3-70b-versatile`
- `qwen/qwen3-32b`
- `openai/gpt-oss-120b`

## Running

Install dependencies, then start Streamlit from the `data_poisoning` folder:

```powershell
pip install -r requirements.txt
streamlit run main.py
```

Add your `GROQ_API_KEY` in the sidebar. You can then:

- Upload PDF resumes into the `cvs` folder through the UI.
- Toggle sanitization and guardrails on or off.
- Choose a Groq model for the LangGraph flow.
- Run a baseline model comparison with both protections disabled.

## Expected behavior

- When malicious PDF content is detected during ingestion, the terminal prints a security alert and the resume page is replaced with a redaction placeholder.
- When guardrails catch a system-override pattern in downstream content, the assistant returns the configured refusal response instead of ranking candidates.

## Disclaimer

This project is for educational purposes only. Use it to study prompt injection and defenses in a controlled setting.
