# GamePulse AI

Generative AI-Powered Video Game Review Explorer

## Purpose

GamePulse AI is a college project for exploring Steam player reviews. It loads a published review sample, cleans it in memory with Pandas, and presents the result in a Streamlit dashboard with filters, Plotly charts, a review explorer, OpenAI review analysis, and a dataset chatbot.

## Technology stack

- Python
- Streamlit
- Pandas
- Plotly
- OpenAI API

API credentials come from the `OPENAI_API_KEY` environment variable or from Streamlit secrets. They are not stored in source code.

## Features

- Pandas loads `data/steam_game_reviews.csv` and cleans a copy in memory. The file on disk is not modified.
- Sidebar filters for game, recommendation, Early Access, and date range apply together to the metrics, charts, and reviews.
- Plotly charts show the recommendation split, review volume by game, reviews over time, and playtime.
- The Player Review Explorer shows recent matching reviews with the original wording.
- AI Review Analysis sends one selected review to OpenAI after the user clicks Analyze Review with AI. The response is a structured reading with sentiment, an overall assessment, a main theme, key points, and a summary. That sentiment is separate from the dataset recommendation label.
- Ask GamePulse answers questions about the currently filtered dataset. Pandas computes the statistics first. OpenAI writes the reply from that context. The full CSV is not sent to the API.

## Dataset

The app uses a published sample of the Steam Reviews Dataset. The original collection and the sample used here are different files.

- Original dataset: Luthfi Mahendra, https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset
- Published 20,000-row sample: https://github.com/mrafifrbbn/steam-reviews/blob/main/data/steam_reviews.csv
- Local file: `data/steam_game_reviews.csv`, that published sample saved unchanged

The Kaggle license was not verified for this project, so this README does not state one. The GitHub repository that published the sample uses the MIT License.

## Generative AI integration

OpenAI analyzes one selected player review at a time. The model returns sentiment, a short assessment, a main theme, key points, and a summary. That output is an AI reading of the review, not a verified fact.

The request includes the selected review plus the game title, hours played, the dataset recommendation, and the Early Access flag. A request is made only after the user clicks Analyze Review with AI.

For local use, set the `OPENAI_API_KEY` environment variable. An optional `OPENAI_MODEL` setting overrides the default model.

## Dataset chatbot

Ask GamePulse answers questions about the reviews currently selected by the dashboard filters. Numeric facts are calculated with Pandas before the request. When a question is about player wording, the request can include a few copied excerpts. The model is instructed to use only that supplied context and to say when the context does not contain an answer. A request is made only after the user submits a question.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

Do not commit an API key. `.gitignore` excludes `.streamlit/secrets.toml`.

Locally, set `OPENAI_API_KEY` in the environment, or put it in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
```

The app checks the environment variable first, then Streamlit secrets.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub. Leave `.streamlit/secrets.toml` untracked.
2. In Community Cloud, create an app from the repository with `app.py` as the entrypoint.
3. In Advanced settings, choose a supported Python version. The default is 3.12. Python 3.10 through 3.14 are supported.
4. Paste the secrets into the Secrets field, using the same `OPENAI_API_KEY` name as the local file.
5. Deploy. The CSV in `data/` is part of the repository, so the app does not need a separate data download.
