"""Dataset question answering for the active filtered reviews.

Statistics are computed with Pandas before any model call. The full CSV and
the full review text are not sent to the API.
"""

from ai_analysis import (
    AnalysisError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
    get_config_value,
    get_model_name,
)

MAX_HISTORY_MESSAGES = 8
MAX_EXCERPTS = 4
MAX_EXCERPT_CHARS = 500
OPINION_HINTS = (
    "saying",
    "say about",
    "opinion",
    "mention",
    "complain",
    "feedback",
    "what are",
    "tell me about",
    "reviews for",
    "talk about",
    "think about",
    "players",
    "people saying",
    "feel about",
)

SYSTEM_PROMPT = (
    "You are GamePulse, a dataset analysis assistant. Answer questions using "
    "only the supplied Steam review dataset context. Never invent statistics, "
    "games, dates, or review details. When a numeric answer can be computed "
    "from the dataset context, use the supplied values. If the information is "
    "unavailable, say so clearly. Distinguish computed dataset facts from "
    "qualitative interpretations. The current context represents the user's "
    "active filters. The dataset field named recommendation is a source label "
    "(Recommended or Not Recommended). It is not an AI sentiment score. "
    "Earlier messages may refer to a previous filter. If they conflict with "
    "the current context, use the current context."
)


def _format_rate(recommended_count, total_count):
    if total_count == 0:
        return "not available"
    return f"{(recommended_count / total_count) * 100:.1f}%"


def _format_hours(value):
    if value is None or value != value:
        return "not available"
    return f"{float(value):.1f}"


def _game_statistics(reviews):
    grouped = reviews.groupby("title", dropna=False)
    rows = []
    for title, group in grouped:
        recommended_count = int((group["recommendation"] == "Recommended").sum())
        not_recommended_count = int((group["recommendation"] == "Not Recommended").sum())
        review_count = int(len(group))
        rows.append(
            {
                "title": "Unknown title" if title is None or title != title else str(title),
                "review_count": review_count,
                "recommended_count": recommended_count,
                "not_recommended_count": not_recommended_count,
                "recommendation_rate": _format_rate(recommended_count, review_count),
                "average_hours": _format_hours(group["hour_played"].mean()),
                "average_hours_value": group["hour_played"].mean(),
            }
        )
    rows.sort(key=lambda item: (-item["review_count"], item["title"]))
    return rows


def _tied_names(rows, value_name):
    if not rows:
        return []
    best = max(item[value_name] for item in rows)
    names = [item["title"] for item in rows if item[value_name] == best]
    return sorted(names), best


def build_dataset_context(reviews):
    """Return a compact text summary computed from the filtered rows."""
    total_count = int(len(reviews))
    game_count = int(reviews["title"].nunique(dropna=True)) if total_count else 0
    recommended_count = int((reviews["recommendation"] == "Recommended").sum()) if total_count else 0
    not_recommended_count = (
        int((reviews["recommendation"] == "Not Recommended").sum()) if total_count else 0
    )
    early_access_count = (
        int(reviews["is_early_access_review"].eq(True).sum()) if total_count else 0
    )
    non_early_access_count = total_count - early_access_count
    average_hours = reviews["hour_played"].mean() if total_count else None
    dated = reviews.dropna(subset=["date_posted"]) if total_count else reviews
    if len(dated):
        minimum_date = dated["date_posted"].min().strftime("%Y-%m-%d")
        maximum_date = dated["date_posted"].max().strftime("%Y-%m-%d")
        year_counts = dated.groupby(dated["date_posted"].dt.year).size()
    else:
        minimum_date = "not available"
        maximum_date = "not available"
        year_counts = None

    lines = [
        "Active filtered dataset context. These values were computed with Pandas.",
        "Use these values directly. Do not invent counts or recalculate them from excerpts.",
        f"Total reviews: {total_count}",
        f"Unique games: {game_count}",
        f"Recommended count: {recommended_count}",
        f"Not recommended count: {not_recommended_count}",
        f"Recommendation rate: {_format_rate(recommended_count, total_count)}",
        f"Average hours played: {_format_hours(average_hours)}",
        f"Early access reviews: {early_access_count}",
        f"Non-early access reviews: {non_early_access_count}",
        f"Minimum review date: {minimum_date}",
        f"Maximum review date: {maximum_date}",
        "Fields: title (game name), review (player text), recommendation "
        "(source label Recommended or Not Recommended, not an AI sentiment), "
        "date_posted, hour_played, helpful (helpful vote count), funny "
        "(funny vote count), is_early_access_review.",
    ]
    if year_counts is not None and len(year_counts):
        year_text = ", ".join(f"{int(year)}: {int(count)}" for year, count in year_counts.items())
        lines.append(f"Review count by year: {year_text}")
    else:
        lines.append("Review count by year: not available")

    game_rows = _game_statistics(reviews) if total_count else []
    if game_rows:
        most_names, most_count = _tied_names(game_rows, "review_count")
        lines.append(
            "Game with the most reviews: "
            + ", ".join(most_names)
            + f" ({int(most_count)} reviews)"
        )
        hour_rows = [
            item for item in game_rows if item["average_hours_value"] == item["average_hours_value"]
        ]
        if hour_rows:
            highest_names, highest_hours = _tied_names(hour_rows, "average_hours_value")
            review_counts = {item["title"]: item["review_count"] for item in hour_rows}
            named = ", ".join(
                f"{name} ({review_counts[name]} reviews)" for name in highest_names
            )
            lines.append(
                "Highest average hours played in this filter: "
                + named
                + f", mean {_format_hours(highest_hours)} hours. "
                "This is the highest mean among games in the current filter, "
                "including games with very few reviews, so it is not a reliability ranking."
            )
        lines.append("Per-game statistics:")
        for item in game_rows:
            lines.append(
                f"- {item['title']}: reviews={item['review_count']}, "
                f"recommended={item['recommended_count']}, "
                f"not_recommended={item['not_recommended_count']}, "
                f"recommendation_rate={item['recommendation_rate']}, "
                f"average_hours={item['average_hours']}"
            )
    else:
        lines.append("Per-game statistics: none. The active filter has no reviews.")
    return "\n".join(lines)


def needs_review_excerpts(question):
    """Return whether the question asks about review wording."""
    text = question.lower()
    return any(hint in text for hint in OPINION_HINTS)


def _question_tokens(question):
    tokens = []
    for raw in question.lower().replace("?", " ").split():
        token = "".join(character for character in raw if character.isalnum())
        if len(token) > 3 and token not in {"what", "this", "that", "with", "from", "about", "have"}:
            tokens.append(token)
    return tokens


def select_review_excerpts(reviews, question):
    """Copy a few relevant reviews. Long reviews are marked when shortened."""
    if reviews.empty or not needs_review_excerpts(question):
        return []
    working = reviews
    question_lower = question.lower()
    mentioned_titles = [
        title
        for title in working["title"].dropna().unique()
        if str(title).lower() in question_lower
    ]
    if mentioned_titles:
        matching = working.loc[working["title"].isin(mentioned_titles)]
        if not matching.empty:
            working = matching

    tokens = _question_tokens(question)
    scored = []
    for position, row in working.reset_index(drop=True).iterrows():
        review_text = row["review"] if isinstance(row["review"], str) else ""
        review_text = review_text.strip()
        if not review_text:
            continue
        lowered = review_text.lower()
        score = sum(1 for token in tokens if token in lowered)
        scored.append((score, position, row, review_text))
    scored.sort(key=lambda item: (-item[0], item[1]))

    excerpts = []
    for _score, _position, row, review_text in scored[:MAX_EXCERPTS]:
        truncated = len(review_text) > MAX_EXCERPT_CHARS
        wording = review_text[:MAX_EXCERPT_CHARS] if truncated else review_text
        excerpts.append(
            {
                "title": row["title"],
                "recommendation": row["recommendation"],
                "hours_played": _format_hours(row["hour_played"]),
                "early_access": "Yes" if bool(row["is_early_access_review"]) else "No",
                "wording": wording,
                "truncated": truncated,
            }
        )
    return excerpts


def _format_excerpts(excerpts):
    if not excerpts:
        return (
            "Relevant review excerpts: none. Do not invent player quotations. "
            "If the question needs review wording that is not included here, "
            "say the dataset context does not provide enough information."
        )
    lines = [
        "Relevant review excerpts. The wording is copied from the dataset. "
        "A cut-off note means the original review continues beyond this excerpt."
    ]
    for number, excerpt in enumerate(excerpts, start=1):
        lines.append(
            f"{number}. Game: {excerpt['title']} | Dataset recommendation: "
            f"{excerpt['recommendation']} | Hours played: {excerpt['hours_played']} | "
            f"Early access: {excerpt['early_access']}"
        )
        lines.append(excerpt["wording"])
        if excerpt["truncated"]:
            lines.append("[Excerpt cut off after 500 characters.]")
    return "\n".join(lines)


def _bounded_history(history):
    messages = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    return messages[-MAX_HISTORY_MESSAGES:]


def answer_dataset_question(question, reviews, history=None):
    """Answer one user question with one Responses API request."""
    if not isinstance(question, str) or not question.strip():
        raise AnalysisError("Enter a question about the filtered reviews.")
    question = question.strip()[:2000]

    api_key = get_config_value("OPENAI_API_KEY")
    if not api_key:
        raise AnalysisError(
            "No OpenAI API key was found. Set OPENAI_API_KEY in the environment "
            "for local use, or add OPENAI_API_KEY to Streamlit secrets for Community Cloud."
        )

    context = build_dataset_context(reviews)
    excerpts = _format_excerpts(select_review_excerpts(reviews, question))
    current_message = "\n\n".join(
        (
            context,
            excerpts,
            f"User question: {question}",
        )
    )
    model_input = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_input.extend(_bounded_history(history))
    model_input.append({"role": "user", "content": current_message})

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=get_model_name(),
            input=model_input,
            max_output_tokens=700,
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

    answer = getattr(response, "output_text", None)
    if not isinstance(answer, str) or not answer.strip():
        raise AnalysisError(
            "The chatbot returned an empty response. Try asking the question again."
        )
    return answer.strip()
