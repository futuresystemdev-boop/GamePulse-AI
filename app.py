import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from ai_analysis import AnalysisError, ReviewAnalysis, analyze_review
from dataset_chat import answer_dataset_question

DATA_PATH = Path(__file__).resolve().parent / "data" / "steam_game_reviews.csv"
REQUIRED_COLUMNS = (
    "title",
    "review",
    "recommendation",
    "date_posted",
    "hour_played",
    "helpful",
    "funny",
    "is_early_access_review",
)
TEXT_COLUMNS = ("title", "review", "recommendation")
NUMERIC_COLUMNS = ("hour_played", "helpful", "funny")
# Keep bar-chart labels readable. Metrics still count every filtered game.
TOP_GAME_BARS = 12


def _strip_text(value):
    """Strip surrounding whitespace without changing nulls or review wording."""
    if isinstance(value, str):
        return value.strip()
    return value


def _has_review_text(value):
    return isinstance(value, str) and value.strip() != ""


@st.cache_data
def load_and_clean_reviews(csv_path):
    """Load the review CSV and clean a copy in memory.

    The file on disk is not written. Rows with no usable review text are left
    out of the returned DataFrame so later text analysis can skip them.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(csv_path)

    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The review CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The review CSV could not be parsed.") from exc
    except UnicodeError as exc:
        raise ValueError("The review CSV could not be read as text.") from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"The review CSV is missing required columns: {names}.")

    raw_row_count = len(raw)
    column_count = raw.shape[1]
    missing_review_count = int((~raw["review"].map(_has_review_text)).sum())

    cleaned = raw.copy()
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].map(_strip_text)

    cleaned["date_posted"] = pd.to_datetime(cleaned["date_posted"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    duplicate_row_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.loc[cleaned["review"].map(_has_review_text)].reset_index(drop=True)

    status = {
        "raw_row_count": raw_row_count,
        "cleaned_row_count": len(cleaned),
        "column_count": column_count,
        "missing_review_count": missing_review_count,
        "duplicate_row_count": duplicate_row_count,
    }
    return cleaned, status


def filter_reviews(reviews, selected_titles, recommendation, early_access, start_date, end_date):
    """Return a new DataFrame with the sidebar filters applied together."""
    filtered = reviews.copy()
    # An empty selection means every game, so the page opens unfiltered.
    if selected_titles:
        filtered = filtered.loc[filtered["title"].isin(selected_titles)]
    if recommendation != "All":
        filtered = filtered.loc[filtered["recommendation"] == recommendation]
    if early_access == "Early Access Reviews":
        filtered = filtered.loc[filtered["is_early_access_review"].eq(True)]
    elif early_access == "Non-Early Access Reviews":
        filtered = filtered.loc[filtered["is_early_access_review"].eq(False)]
    if start_date is not None and end_date is not None:
        posted = filtered["date_posted"]
        filtered = filtered.loc[
            posted.notna()
            & (posted.dt.date >= start_date)
            & (posted.dt.date <= end_date)
        ]
    return filtered.copy()


def recommendation_chart(filtered):
    counts = filtered["recommendation"].value_counts()
    counts = counts.reindex(["Recommended", "Not Recommended"]).dropna().astype(int)
    counts = counts.loc[counts > 0]
    figure = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.55,
        title="Recommended and not recommended reviews",
        color=counts.index,
        color_discrete_map={
            "Recommended": "#3DDC97",
            "Not Recommended": "#FF6B6B",
        },
    )
    figure.update_layout(margin=dict(l=16, r=16, t=56, b=16), legend_title_text="")
    return figure


def game_volume_chart(filtered):
    counts = (
        filtered.groupby("title", as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
        .sort_values(["reviews", "title"], ascending=[False, True])
    )
    shown = counts.head(TOP_GAME_BARS).sort_values("reviews", ascending=True)
    title = "Reviews by game"
    if len(counts) > len(shown):
        title = f"Top {len(shown)} games by review count"
    figure = px.bar(shown, x="reviews", y="title", orientation="h", title=title)
    figure.update_layout(
        xaxis_title="Reviews",
        yaxis_title=None,
        height=max(360, 42 * len(shown) + 90),
        margin=dict(l=8, r=16, t=56, b=8),
    )
    figure.update_yaxes(automargin=True)
    return figure, len(counts), len(shown)


def reviews_over_time_chart(filtered):
    dated = filtered.dropna(subset=["date_posted"])
    if dated.empty:
        return None
    monthly = (
        dated.assign(month=dated["date_posted"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
    )
    monthly.index.name = "month"
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0)
    monthly.index.name = "month"
    chart_data = monthly.rename("reviews").reset_index()
    figure = px.line(
        chart_data,
        x="month",
        y="reviews",
        markers=True,
        title="Reviews posted each month",
    )
    figure.update_layout(
        xaxis_title=None,
        yaxis_title="Reviews",
        margin=dict(l=8, r=16, t=56, b=8),
    )
    return figure


def playtime_chart(filtered):
    playtime = filtered["hour_played"].dropna()
    if playtime.empty:
        return None
    figure = px.histogram(
        x=playtime,
        nbins=40,
        title="Hours played at the time of review",
    )
    figure.update_layout(
        xaxis_title="Hours played",
        yaxis_title="Reviews",
        margin=dict(l=8, r=16, t=56, b=8),
        showlegend=False,
    )
    return figure


def _format_review_date(value):
    if pd.isna(value):
        return "Unknown date"
    return value.strftime("%b %d, %Y")


def _format_hours(value):
    if pd.isna(value):
        return "Unknown"
    return f"{value:,.0f} hours"


def _review_text(value):
    if isinstance(value, str) and value.strip():
        return value
    return "No review text is available for this row."


def _show_ai_result(result):
    with st.container(border=True):
        sentiment_column, theme_column = st.columns(2)
        sentiment_column.markdown("**AI Sentiment**")
        sentiment_column.write(result.sentiment)
        theme_column.markdown("**Main Theme**")
        theme_column.write(result.main_theme)
        st.markdown("**Overall Assessment**")
        st.write(result.overall_assessment)
        st.markdown("**Key Points**")
        for point in result.key_points:
            st.write(f"• {point}")
        st.markdown("**Summary**")
        st.write(result.summary)
        st.caption("AI-generated analysis based on the selected player review.")


def _review_analysis_id(title, review_text):
    return hashlib.sha1(f"{title}|{review_text}".encode()).hexdigest()


def _render_ai_analysis(preview):
    """Show the analysis controls. The API is called only from the button."""
    st.subheader("AI Review Analysis")
    if preview is None or preview.empty:
        st.info("No reviews match these filters, so there is nothing to analyze.")
        return

    st.session_state.setdefault("ai_results", {})
    st.session_state.setdefault("ai_errors", {})

    options = []
    rows_by_label = {}
    for position, row in enumerate(preview.itertuples(index=False)):
        snippet = " ".join(str(row.review).split())
        if len(snippet) > 70:
            snippet = snippet[:67] + "..."
        label = (
            f"{position + 1}. {_format_review_date(row.date_posted)} — "
            f"{row.title} — {snippet}"
        )
        options.append(label)
        rows_by_label[label] = row

    saved_choice = st.session_state.get("ai_review_choice")
    if saved_choice not in options:
        st.session_state["ai_review_choice"] = options[0]
    selected_label = st.selectbox("Review to analyze", options, key="ai_review_choice")
    selected = rows_by_label[selected_label]
    review_text = selected.review if isinstance(selected.review, str) else ""
    review_id = _review_analysis_id(selected.title, review_text)

    st.caption(
        f"{selected.title} · {_format_hours(selected.hour_played)} · "
        f"{selected.recommendation}"
    )
    st.caption("Only this review is sent when you click the button.")

    if st.button("Analyze Review with AI", type="primary"):
        with st.spinner("Analyzing the selected review..."):
            try:
                result = analyze_review(
                    selected.title,
                    review_text,
                    selected.hour_played,
                    selected.recommendation,
                    selected.is_early_access_review,
                )
            except AnalysisError as error:
                st.session_state["ai_errors"][review_id] = error.message
                st.session_state["ai_results"].pop(review_id, None)
            else:
                st.session_state["ai_errors"].pop(review_id, None)
                st.session_state["ai_results"][review_id] = result.model_dump()

    error_message = st.session_state["ai_errors"].get(review_id)
    saved_result = st.session_state["ai_results"].get(review_id)
    if error_message:
        st.error(error_message)
    elif saved_result:
        try:
            _show_ai_result(ReviewAnalysis.model_validate(saved_result))
        except ValidationError:
            st.error(
                "The AI response did not match the expected analysis format. Try again."
            )


st.set_page_config(
    page_title="GamePulse AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GamePulse AI")
st.write("Explore Steam player reviews with data analytics and Generative AI.")

try:
    reviews, status = load_and_clean_reviews(str(DATA_PATH))
except FileNotFoundError:
    st.error(
        "The review dataset was not found at data/steam_game_reviews.csv. "
        "Add that file and reload the app."
    )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except OSError:
    st.error(
        "The review dataset could not be opened. Check that "
        "data/steam_game_reviews.csv is readable."
    )
    st.stop()

dated_reviews = reviews.dropna(subset=["date_posted"])
if dated_reviews.empty:
    min_date = None
    max_date = None
else:
    min_date = dated_reviews["date_posted"].min().date()
    max_date = dated_reviews["date_posted"].max().date()

game_options = sorted(reviews["title"].dropna().unique())

st.sidebar.header("Explore Filters")
st.sidebar.caption("These filters apply together to the metrics, charts, and reviews.")
selected_titles = st.sidebar.multiselect(
    "Game",
    options=game_options,
    placeholder="All games",
    help="Leave this empty to include every game.",
)
recommendation = st.sidebar.selectbox(
    "Recommendation",
    ("All", "Recommended", "Not Recommended"),
)
early_access = st.sidebar.selectbox(
    "Early Access",
    ("All", "Early Access Reviews", "Non-Early Access Reviews"),
)
if min_date is None or max_date is None:
    start_date = None
    end_date = None
    st.sidebar.caption("No review dates are available to filter.")
else:
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Limit reviews to the dates they were posted.",
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = sorted(date_range)
    else:
        start_date, end_date = min_date, max_date
        st.sidebar.info("Select a start and end date to apply the date filter.")

filtered = filter_reviews(
    reviews,
    selected_titles,
    recommendation,
    early_access,
    start_date,
    end_date,
)

total_reviews = len(filtered)
games_represented = int(filtered["title"].nunique())
if total_reviews == 0:
    recommendation_rate = None
    average_hours = None
else:
    recommendation_rate = (filtered["recommendation"] == "Recommended").mean() * 100
    average_hours = filtered["hour_played"].mean()

st.subheader("Overview")
metric_columns = st.columns(4)
metric_columns[0].metric("Total Reviews", f"{total_reviews:,}")
metric_columns[1].metric("Games Represented", f"{games_represented:,}")
metric_columns[2].metric(
    "Recommendation Rate",
    "—" if recommendation_rate is None else f"{recommendation_rate:.1f}%",
)
metric_columns[3].metric(
    "Average Hours Played",
    "—" if average_hours is None or pd.isna(average_hours) else f"{average_hours:,.1f} hrs",
)

if filtered.empty:
    st.warning(
        "No reviews match these filters. Try a wider date range, or choose a "
        "different game, recommendation, or Early Access option."
    )
    _render_ai_analysis(None)
else:
    distribution_column, playtime_column = st.columns(2)
    with distribution_column:
        st.plotly_chart(
            recommendation_chart(filtered),
            width="stretch",
            theme="streamlit",
        )
    with playtime_column:
        playtime_figure = playtime_chart(filtered)
        if playtime_figure is None:
            st.info("No playtime values are available for this filter.")
        else:
            st.plotly_chart(playtime_figure, width="stretch", theme="streamlit")
            playtime = filtered["hour_played"].dropna()
            st.caption(
                f"Minimum {playtime.min():,.0f} hours, median {playtime.median():,.0f} hours, "
                f"maximum {playtime.max():,.0f} hours. All playtime values are included."
            )

    volume_figure, game_count, games_shown = game_volume_chart(filtered)
    st.plotly_chart(volume_figure, width="stretch", theme="streamlit")
    if game_count > games_shown:
        st.caption(
            f"Showing the {games_shown} games with the most reviews. "
            "The other games remain in the metrics and the review explorer."
        )

    timeline = reviews_over_time_chart(filtered)
    if timeline is None:
        st.info("No review dates are available for this filter.")
    else:
        st.plotly_chart(timeline, width="stretch", theme="streamlit")

    st.subheader("Player Review Explorer")
    st.caption("Most recent matching reviews. The wording is shown as written.")
    preview_count = st.slider("Reviews to show", min_value=3, max_value=15, value=5)
    preview = filtered.sort_values(
        ["date_posted", "title"],
        ascending=[False, True],
    ).head(preview_count)

    for position, row in enumerate(preview.itertuples(index=False)):
        with st.container(border=True):
            game_column, date_column, recommendation_column, hours_column = st.columns(4)
            game_column.markdown("**Game**")
            game_column.write(row.title)
            date_column.markdown("**Date**")
            date_column.write(_format_review_date(row.date_posted))
            recommendation_column.markdown("**Recommendation**")
            recommendation_column.write(row.recommendation)
            hours_column.markdown("**Hours Played**")
            hours_column.write(_format_hours(row.hour_played))
            review_text = _review_text(row.review)
            # The key includes the review so a new filter cannot keep the previous text.
            review_key = hashlib.sha1(
                f"{position}|{row.title}|{review_text}".encode()
            ).hexdigest()
            st.text_area(
                "Review text",
                value=review_text,
                height=140,
                disabled=True,
                label_visibility="collapsed",
                key=f"review_{review_key}",
            )

    _render_ai_analysis(preview)

with st.expander("Dataset Information"):
    st.markdown(
        "\n".join(
            (
                f"- Raw dataset rows: {status['raw_row_count']:,}",
                f"- Usable review rows: {status['cleaned_row_count']:,}",
                f"- Number of games: {reviews['title'].nunique():,}",
                f"- Missing review count: {status['missing_review_count']:,}",
            )
        )
    )
    st.caption("The CSV file is not modified. Cleaning and filtering happen in memory.")


def _render_ask_gamepulse(filtered_reviews):
    """Show the dataset chatbot. A request is sent only when a question is submitted."""
    st.subheader("Ask GamePulse")
    st.write("Ask questions about the currently filtered Steam review dataset.")
    st.session_state.setdefault("gamepulse_messages", [])
    # A cancelled run can leave a question with no reply. Drop that orphan
    # instead of sending it to the API on a later rerun.
    if (
        st.session_state.gamepulse_messages
        and st.session_state.gamepulse_messages[-1]["role"] == "user"
    ):
        st.session_state.gamepulse_messages.pop()

    with st.chat_message("assistant"):
        st.write(
            "Ask me about the reviews, games, recommendations, playtime, dates, "
            "or other information available in the dataset."
        )
    for message in st.session_state.gamepulse_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about the filtered reviews")
    if not isinstance(question, str) or not question.strip():
        return

    prior_messages = list(st.session_state.gamepulse_messages)
    try:
        with st.spinner("Checking the filtered dataset..."):
            answer = answer_dataset_question(question, filtered_reviews, prior_messages)
    except AnalysisError as error:
        answer = error.message
    except Exception:
        answer = "The chatbot could not complete that request. Try again in a moment."
    st.session_state.gamepulse_messages.append(
        {"role": "user", "content": question.strip()}
    )
    st.session_state.gamepulse_messages.append({"role": "assistant", "content": answer})
    if len(st.session_state.gamepulse_messages) > 12:
        st.session_state.gamepulse_messages = st.session_state.gamepulse_messages[-12:]
    st.rerun()


_render_ask_gamepulse(filtered)
