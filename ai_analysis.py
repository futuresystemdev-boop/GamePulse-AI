"""OpenAI settings and single-review analysis.

The API key is read when an analysis is requested. It is not written to disk,
logged, or returned in error messages.
"""

import os
from typing import Literal

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

DEFAULT_MODEL = "gpt-6-astra"
SENTIMENTS = ("Positive", "Negative", "Mixed", "Neutral")


class AnalysisError(Exception):
    """A user-facing analysis failure that should not crash the app."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ReviewAnalysis(BaseModel):
    """Structured qualitative analysis of one player review."""

    sentiment: Literal["Positive", "Negative", "Mixed", "Neutral"]
    overall_assessment: str
    main_theme: str
    key_points: list[str] = Field(min_length=2, max_length=4)
    summary: str


def get_config_value(name):
    """Read one setting from the environment, then from Streamlit secrets."""
    environment_value = os.environ.get(name)
    if isinstance(environment_value, str) and environment_value.strip():
        return environment_value.strip()
    try:
        if name in st.secrets:
            secret_value = st.secrets[name]
            if secret_value is None:
                return None
            text = str(secret_value).strip()
            return text or None
    except Exception:
        return None
    return None


def get_model_name():
    """Return the configured model, or the current default."""
    return get_config_value("OPENAI_MODEL") or DEFAULT_MODEL


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(value != value)
    except TypeError:
        return False


def _hours_label(hours_played):
    if _is_missing(hours_played):
        return "Not available"
    number = float(hours_played)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _early_access_label(early_access):
    if early_access is None or _is_missing(early_access):
        return "Not available"
    return "Yes" if bool(early_access) else "No"


def build_analysis_input(title, review_text, hours_played, recommendation, early_access):
    """Build the Responses API input for one review."""
    system_prompt = (
        "You analyze one Steam player review. Use only the supplied review and "
        "the short context fields. Do not invent facts about the game, its "
        "developers, other players, or events that the review does not state. "
        "The dataset recommendation is source-file context. Do not treat that "
        "label as automatically deciding sentiment. Choose sentiment "
        "independently from Positive, Negative, Mixed, or Neutral. Use Mixed "
        "or Neutral when the review is ambiguous or balanced. The main theme "
        "should be a short category grounded in the review, such as Gameplay, "
        "Performance, Bugs, Graphics, Multiplayer, Content, Value, or another "
        "fitting phrase. Return 2 to 4 key points that are supported by the "
        "review text."
    )
    recommendation_label = (
        "Not available" if _is_missing(recommendation) else str(recommendation).strip()
    )
    user_prompt = "\n".join(
        (
            f"Game title: {title if not _is_missing(title) else 'Not available'}",
            f"Hours played: {_hours_label(hours_played)}",
            f"Dataset recommendation: {recommendation_label}",
            f"Early access review: {_early_access_label(early_access)}",
            "Review text:",
            review_text,
        )
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_analysis(analysis):
    points = [point.strip() for point in analysis.key_points if point and point.strip()]
    normalized = analysis.model_copy(
        update={
            "overall_assessment": analysis.overall_assessment.strip(),
            "main_theme": analysis.main_theme.strip(),
            "summary": analysis.summary.strip(),
            "key_points": points,
        }
    )
    if normalized.sentiment not in SENTIMENTS:
        raise AnalysisError(
            "The AI response did not match the expected analysis format. Try again."
        )
    if not (
        normalized.overall_assessment
        and normalized.main_theme
        and normalized.summary
        and 2 <= len(normalized.key_points) <= 4
    ):
        raise AnalysisError(
            "The AI response did not match the expected analysis format. Try again."
        )
    return normalized


def analyze_review(title, review_text, hours_played, recommendation, early_access):
    """Send one review to the OpenAI Responses API and validate the result."""
    if not isinstance(review_text, str) or not review_text.strip():
        raise AnalysisError(
            "This review has no text to analyze. Choose a review that includes written feedback."
        )

    api_key = get_config_value("OPENAI_API_KEY")
    if not api_key:
        raise AnalysisError(
            "No OpenAI API key was found. Set OPENAI_API_KEY in the environment "
            "for local use, or add OPENAI_API_KEY to Streamlit secrets for Community Cloud."
        )

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.parse(
            model=get_model_name(),
            input=build_analysis_input(
                title,
                review_text,
                hours_played,
                recommendation,
                early_access,
            ),
            text_format=ReviewAnalysis,
        )
    except AuthenticationError:
        raise AnalysisError(
            "The OpenAI API key was rejected. Check the key and try again."
        ) from None
    except RateLimitError:
        raise AnalysisError(
            "The OpenAI API rate limit was reached. Wait a moment and try again."
        ) from None
    except APIConnectionError:
        raise AnalysisError(
            "The app could not reach the OpenAI API. Check the connection and try again."
        ) from None
    except APIStatusError:
        raise AnalysisError(
            "The OpenAI API returned an error. Try again in a moment."
        ) from None
    except ValidationError:
        raise AnalysisError(
            "The AI response did not match the expected analysis format. Try again."
        ) from None

    parsed = getattr(response, "output_parsed", None)
    if not isinstance(parsed, ReviewAnalysis):
        raise AnalysisError(
            "The AI response did not match the expected analysis format. Try again."
        )
    return _normalize_analysis(parsed)
