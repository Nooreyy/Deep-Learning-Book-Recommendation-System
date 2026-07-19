"""Streamlit interface for the Deep Learning Book Recommendation System."""

from __future__ import annotations

import base64
import html
from typing import Any

import pandas as pd
import streamlit as st

import backend


st.set_page_config(
    page_title="Deep Learning Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp { background: #0F172A; color: #FFFFFF; }
    [data-testid="stSidebar"] { background: #111C31; border-right: 1px solid #263650; }
    [data-testid="stSidebar"] * { color: #E2E8F0; }
    .block-container { max-width: 1450px; padding-top: 2.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 3.25rem 2.75rem; border-radius: 16px; color: #FFFFFF;
        background: linear-gradient(118deg, #0B3B62 0%, #1E293B 45%, #49267A 100%);
        box-shadow: 0 18px 45px rgba(0, 0, 0, .28); margin-bottom: 1.8rem;
    }
    .hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 800; letter-spacing: -.04em; }
    .hero p { margin: .85rem 0 0; color: #CBD5E1; font-size: 1.15rem; max-width: 760px; line-height: 1.6; }
    .section-title { color: #F8FAFC; font-size: 1.45rem; font-weight: 750; margin: .5rem 0 1rem; }
    .book-card {
        min-height: 490px; overflow: hidden; border: 1px solid rgba(148,163,184,.18);
        background: #1E293B; border-radius: 16px; box-shadow: 0 10px 28px rgba(0,0,0,.18);
        transition: transform .22s ease, box-shadow .22s ease; margin-bottom: 1.2rem;
    }
    .book-card:hover { transform: translateY(-6px); box-shadow: 0 18px 36px rgba(56,189,248,.14); }
    .book-cover { width: 100%; height: 245px; object-fit: cover; display: block; background: #0F172A; }
    .book-content { padding: 1.2rem 1.25rem 1.35rem; }
    .book-title { color: #FFFFFF; font-size: 1.12rem; font-weight: 750; line-height: 1.35; min-height: 3.05rem; }
    .book-author { color: #38BDF8; font-size: .95rem; margin: .5rem 0 .85rem; }
    .book-meta { color: #CBD5E1; font-size: .85rem; line-height: 1.6; min-height: 2.8rem; }
    .star-rating { color: #FBBF24; font-size: 1rem; margin-top: 1rem; letter-spacing: 1px; }
    .empty-state {
        text-align: center; padding: 4.5rem 1rem; border: 1px dashed #475569;
        border-radius: 16px; color: #CBD5E1; background: rgba(30,41,59,.55); font-size: 1.1rem;
    }
    .footer {
        margin-top: 3.5rem; padding: 2.3rem 1rem; text-align: center; color: #CBD5E1;
        border-top: 1px solid #334155; line-height: 1.9;
    }
    .footer strong { color: #FFFFFF; }
    .footer .tools { color: #38BDF8; }
    .stButton > button {
        width: 100%; border: 0; border-radius: 12px; padding: .78rem 1rem;
        background: linear-gradient(90deg, #38BDF8, #8B5CF6); color: white; font-weight: 750;
        font-size: 1.05rem; transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover {
        color: white; transform: translateY(-2px); box-shadow: 0 10px 20px rgba(56,189,248,.25);
    }
    [data-testid="stMetric"] {
        background: rgba(15,23,42,.62); border: 1px solid rgba(148,163,184,.16);
        border-radius: 12px; padding: .55rem .75rem;
    }
    [data-testid="stMetricLabel"] { color: #CBD5E1; }
    [data-testid="stMetricValue"] { color: #38BDF8; }
    @media (max-width: 768px) {
        .block-container { padding: 1.25rem 1rem; }
        .hero { padding: 2.2rem 1.5rem; }
    }
</style>
"""


def placeholder_cover() -> str:
    """Return an inline placeholder cover that works without a network request."""
    svg = """<svg xmlns='http://www.w3.org/2000/svg' width='400' height='600'>
    <rect width='100%' height='100%' fill='#0F172A'/>
    <rect x='34' y='34' width='332' height='532' rx='18' fill='#1E293B' stroke='#38BDF8'/>
    <text x='200' y='270' text-anchor='middle' fill='#38BDF8' font-size='74'>📚</text>
    <text x='200' y='342' text-anchor='middle' fill='#CBD5E1' font-size='24'>Book Cover</text></svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


@st.cache_resource(show_spinner=False)
def load_application_resources() -> tuple[Any, dict[str, Any], Any, list[str]]:
    """Load the model and fitted preprocessing objects once per application session."""
    model = backend.load_model()
    encoders, scaler, feature_columns = backend.load_preprocessing_objects()
    return model, encoders, scaler, feature_columns


@st.cache_data(show_spinner=False)
def load_application_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source datasets once and share them safely across reruns."""
    return backend.load_datasets()


def render_sidebar() -> int:
    """Render project information and return the selected recommendation count."""
    with st.sidebar:
        st.title("📖 Project Overview")
        st.caption("An AI-powered portfolio project for personalized discovery.")
        st.divider()

        st.subheader("Model")
        st.write("Feedforward Neural Network (FNN)")
        st.subheader("Dataset")
        st.write("Book-Crossing Dataset")

        st.subheader("Evaluation Metrics")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("MAE", "1.3514")
        metric_right.metric("RMSE", "1.7268")
        st.metric("MSE", "2.9818")

        st.divider()
        st.subheader("Recommendation Slider")
        top_n = st.slider("Top Recommendations", min_value=5, max_value=20, value=9)

        st.divider()
        st.subheader("About Project")
        st.caption(
            "Predictions are generated with TensorFlow/Keras and the same "
            "preprocessing pipeline used during training."
        )
        st.subheader("Developer Information")
        st.write("Designed by Noor Fatima")

    return top_n


def find_cover_url(isbn: str, books_df: pd.DataFrame) -> str:
    """Return the best available cover image URL for a recommended book."""
    try:
        details = backend.get_book_details(isbn, books_df)
        if details:
            for column in ("Image-URL-L", "Image-URL-M", "Image-URL-S"):
                image_url = details.get(column)
                if image_url and str(image_url).strip():
                    return str(image_url)
    except RuntimeError:
        pass

    return placeholder_cover()


def display_book_card(book: pd.Series, books_df: pd.DataFrame) -> None:
    """Render one polished recommendation card with safe HTML text values."""
    cover_url = html.escape(find_cover_url(book["ISBN"], books_df), quote=True)
    title = html.escape(str(book["Book-Title"]))
    author = html.escape(str(book["Book-Author"]))
    publisher = html.escape(str(book["Publisher"]))
    year = html.escape(str(book["Year-Of-Publication"]))
    rating = float(book["Predicted_Rating"])
    star_count = max(1, min(5, round(rating / 2)))
    stars = "★" * star_count + "☆" * (5 - star_count)

    st.markdown(
        f"""
        <article class="book-card">
            <img class="book-cover" src="{cover_url}" alt="Cover of {title}">
            <div class="book-content">
                <div class="book-title">{title}</div>
                <div class="book-author">by {author}</div>
                <div class="book-meta"><strong>Publisher:</strong> {publisher}<br>
                <strong>Published:</strong> {year}</div>
                <div class="star-rating">{stars}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Predicted Rating", f"{rating:.2f} / 10")


def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        """
        <footer class="footer">
            <strong>Deep Learning Book Recommendation System</strong><br>
            <span>Built With</span><br>
            <span class="tools">TensorFlow/Keras &nbsp;•&nbsp; Feedforward Neural Network
            &nbsp;•&nbsp; Scikit-learn &nbsp;•&nbsp; Streamlit</span><br>
            Designed by Noor Fatima
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit application and handle user-facing errors gracefully."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <section class="hero">
            <h1>📚 Deep Learning Book Recommendation System</h1>
            <p>AI-powered personalized book recommendations using Feedforward Neural
            Networks (TensorFlow/Keras)</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    top_n = render_sidebar()

    try:
        model, encoders, scaler, feature_columns = load_application_resources()
        books_df, ratings_df, users_df = load_application_datasets()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        st.error(f"The application resources could not be loaded: {error}")
        render_footer()
        return

    st.markdown('<div class="section-title">Select User</div>', unsafe_allow_html=True)

    if "User-ID" not in users_df.columns or users_df.empty:
        st.error("The Users dataset is missing a valid 'User-ID' column.")
        render_footer()
        return

    user_ids = users_df["User-ID"].dropna().drop_duplicates().tolist()
    selected_user = st.selectbox(
        "Search or select a user ID",
        options=user_ids,
        index=None,
        placeholder="Start typing a user ID...",
    )

    if st.button("📖 Recommend Books", disabled=selected_user is None):
        with st.spinner("Generating personalized recommendations..."):
            try:
                recommendations = backend.recommend_books(
                    user_id=selected_user,
                    model=model,
                    books_df=books_df,
                    ratings_df=ratings_df,
                    users_df=users_df,
                    encoders=encoders,
                    scaler=scaler,
                    feature_columns=feature_columns,
                    top_n=top_n,
                )
            except RuntimeError as error:
                st.error(f"Recommendations could not be generated: {error}")
                render_footer()
                return

        if recommendations.empty:
            st.markdown(
                '<div class="empty-state">No recommendations found for this user.</div>',
                unsafe_allow_html=True,
            )
            st.info("Try selecting another user or verify that the source datasets are complete.")
        else:
            st.success(f"Generated {len(recommendations)} personalized book recommendations.")
            st.markdown(
                '<div class="section-title">Recommended for You</div>',
                unsafe_allow_html=True,
            )

            for start in range(0, len(recommendations), 3):
                columns = st.columns(3)

                for column, (_, book) in zip(
                    columns,
                    recommendations.iloc[start:start + 3].iterrows(),
                ):
                    with column:
                        display_book_card(book, books_df)
    else:
        st.info("Choose a user ID to generate tailored book recommendations.")

    render_footer()


if __name__ == "__main__":
    main()