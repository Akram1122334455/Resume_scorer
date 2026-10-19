# AI Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and gives concrete improvements, powered by Google Gemini Flash models.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a tailored keyword match
- **ATS score (0-100)** computed from five weighted categories:
  keywords (30%), content quality (25%), formatting/parsability (20%), structure (15%), readability (10%)
- Strengths, prioritized issues with fixes, present/missing keywords
- Rewrite suggestions for weak bullet points and a step-by-step action plan
- Download the report as Markdown
- Automatic fallback between Gemini Flash models

> The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then provide it in one of three ways:

1. Paste it in the app sidebar, **or**
2. Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), **or**
3. Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```

Start the app:

```bash
streamlit run app.py
```

## Choosing a model

The sidebar lists several Gemini Flash model IDs. If the selected one is unavailable for your key, the app tries the others automatically. To update the list as Google releases new models, edit `MODEL_OPTIONS` at the top of `app.py`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings -> Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Push to GitHub

```bash
git init
printf "venv/\n__pycache__/\n.streamlit/secrets.toml\n.env\n" > .gitignore
git add .
git commit -m "Initial commit: AI resume ATS checker"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Create the empty repo on github.com first (no README/.gitignore). **Never commit your API key.**

## Project structure

```
app.py            # Streamlit app
requirements.txt  # Dependencies
README.md
```

## Notes

- Scanned/image-only resumes can't be read; export a text-based PDF or DOCX.
- Resumes are sent to the Gemini API for analysis; the app does not store them.
- Free-tier Gemini keys have rate limits; retry after a minute if you hit one.
