"""Backend utilities for the Streamlit Deep Learning Book Recommendation System.

This module centralizes artifact and dataset loading. Recommendation behavior is
intentionally left as placeholders until the Streamlit user flow is finalized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as keras_load_model


# Keep all required artifacts relative to this backend file so the application
# works regardless of the directory from which Streamlit is launched.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "book_recommender.keras"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
ENCODERS_PATH = BASE_DIR / "feature_encoders.pkl"
SCALER_PATH = BASE_DIR / "feature_scaler.pkl"
BOOKS_PATH = BASE_DIR / "Books.csv"
RATINGS_PATH = BASE_DIR / "Ratings.csv"
USERS_PATH = BASE_DIR / "Users.csv"


def load_model() -> Any:
    """Load and return the trained Keras recommendation model safely.

    Raises:
        FileNotFoundError: If the saved model file is unavailable.
        RuntimeError: If Keras cannot load the saved model.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    try:
        return keras_load_model(MODEL_PATH)
    except Exception as error:
        raise RuntimeError(f"Unable to load the Keras model: {error}") from error


def load_preprocessing_objects() -> tuple[dict[str, Any], Any, list[str]]:
    """Load fitted encoders, scaler, and the model's expected feature order.

    These objects must be reused at inference time to keep Streamlit inputs
    consistent with the transformations used during model training.
    """
    required_files = {
        "feature encoders": ENCODERS_PATH,
        "feature scaler": SCALER_PATH,
        "feature columns": FEATURE_COLUMNS_PATH,
    }

    missing_files = [
        f"{name} ({path.name})"
        for name, path in required_files.items()
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing preprocessing artifact(s): " + ", ".join(missing_files)
        )

    try:
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    except Exception as error:
        raise RuntimeError(f"Unable to load preprocessing objects: {error}") from error

    if not isinstance(encoders, dict):
        raise TypeError("The saved feature encoders must be stored as a dictionary.")
    if not isinstance(feature_columns, list):
        raise TypeError("The saved feature columns must be stored as a list.")

    return encoders, scaler, feature_columns


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return the Books, Ratings, and Users CSV datasets safely."""
    dataset_paths = {
        "Books": BOOKS_PATH,
        "Ratings": RATINGS_PATH,
        "Users": USERS_PATH,
    }

    missing_files = [
        f"{name} ({path.name})"
        for name, path in dataset_paths.items()
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("Missing dataset file(s): " + ", ".join(missing_files))

    try:
        books_df = pd.read_csv(BOOKS_PATH, low_memory=False)
        ratings_df = pd.read_csv(RATINGS_PATH, low_memory=False)
        users_df = pd.read_csv(USERS_PATH, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Unable to load one or more datasets: {error}") from error

    return books_df, ratings_df, users_df


def preprocess_input(
    input_data: dict[str, Any],
    encoders: dict[str, Any],
    scaler: Any,
    feature_columns: list[str],
) -> np.ndarray:
    """Encode, order, and scale one Streamlit recommendation input sample.

    Unseen categorical values are encoded as ``-1``. This preserves a stable
    numeric input shape without changing the fitted LabelEncoders at runtime.
    """
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a dictionary containing one input sample.")

    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    missing_features = [
        column for column in feature_columns if column not in input_data
    ]
    if missing_features:
        raise ValueError(
            "Missing required input feature(s): " + ", ".join(missing_features)
        )

    try:
        processed_sample: dict[str, float] = {}

        # Process columns in the exact order expected by the trained model.
        for column in feature_columns:
            value = input_data[column]

            # Apply the saved LabelEncoder when this column was categorical
            # during training. Unknown categories receive the reserved value -1.
            if column in encoders:
                encoder = encoders[column]
                category_to_code = {
                    str(category): code
                    for code, category in enumerate(encoder.classes_)
                }
                processed_sample[column] = float(
                    category_to_code.get(str(value), -1)
                )
            else:
                # Non-encoded columns are expected to be numerical features.
                processed_sample[column] = float(value)

        # A one-row DataFrame preserves column names and order for the scaler.
        input_frame = pd.DataFrame([processed_sample], columns=feature_columns)
        scaled_input = scaler.transform(input_frame)

        return np.asarray(scaled_input, dtype=np.float32)

    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise ValueError(f"Unable to preprocess recommendation input: {error}") from error
    
def preprocess_batch(
    input_df: pd.DataFrame,
    encoders: dict[str, Any],
    scaler: Any,
    feature_columns: list[str],
) -> np.ndarray:
    """
    Encode and scale multiple recommendation samples at once.

    This function is optimized for batch inference. Instead of processing
    one book at a time, it transforms the entire candidate dataset in a
    single operation before sending it to the neural network.

    Parameters
    ----------
    input_df : pd.DataFrame
        Candidate recommendation records.

    encoders : dict
        Saved LabelEncoders.

    scaler : StandardScaler
        Saved feature scaler.

    feature_columns : list
        Model input feature order.

    Returns
    -------
    np.ndarray
        Scaled feature matrix ready for model.predict().
    """

    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")

    if input_df.empty:
        raise ValueError("input_df cannot be empty.")

    try:

        # --------------------------------------------------
        # Keep only model input columns
        # --------------------------------------------------
        processed_df = input_df[feature_columns].copy()

        # --------------------------------------------------
        # Encode categorical columns
        # --------------------------------------------------
        for column, encoder in encoders.items():

            if column not in processed_df.columns:
                continue

            mapping = {
                str(value): index
                for index, value in enumerate(encoder.classes_)
            }

            processed_df[column] = (
                processed_df[column]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(float)
            )

        # --------------------------------------------------
        # Convert remaining numerical columns
        # --------------------------------------------------
        for column in feature_columns:

            if column not in encoders:

                processed_df[column] = pd.to_numeric(
                    processed_df[column],
                    errors="coerce"
                )

        # --------------------------------------------------
        # Replace missing numeric values
        # --------------------------------------------------
        processed_df = processed_df.fillna(0)

        # --------------------------------------------------
        # Scale all samples together
        # --------------------------------------------------
        scaled_features = scaler.transform(processed_df)

        return np.asarray(
            scaled_features,
            dtype=np.float32
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to preprocess batch input: {error}"
        ) from error
    
def predict_rating(model: Any, processed_input: np.ndarray) -> float:
    """Predict and return one book rating constrained to the 0--10 scale."""
    if model is None:
        raise ValueError("A loaded Keras model is required for prediction.")

    if not isinstance(processed_input, np.ndarray) or processed_input.size == 0:
        raise ValueError("processed_input must be a non-empty NumPy array.")

    try:
        # Suppress Keras progress output because Streamlit handles user feedback.
        prediction = model.predict(processed_input, verbose=0)

        # The regression model produces one output value for a single sample.
        predicted_rating = float(np.asarray(prediction).reshape(-1)[0])

        # Keep predictions within the valid book-rating scale used by the project.
        return float(np.clip(predicted_rating, 0.0, 10.0))

    except (AttributeError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError(f"Unable to predict the book rating: {error}") from error


def recommend_books(
    user_id: Any,
    model: Any,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame,
    encoders: dict[str, Any],
    scaler: Any,
    feature_columns: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate Top-N book recommendations using batch prediction.
    """

    try:

        # --------------------------------------------------
        # Validate user
        # --------------------------------------------------
        user_record = users_df.loc[
            users_df["User-ID"] == user_id
        ]

        if user_record.empty:
            raise ValueError(f"User {user_id} not found.")

        user_record = user_record.iloc[0]

        # --------------------------------------------------
        # Books already rated
        # --------------------------------------------------
        rated_books = ratings_df.loc[
            ratings_df["User-ID"] == user_id,
            "ISBN"
        ].astype(str)

        # --------------------------------------------------
        # Candidate books
        # --------------------------------------------------
        candidate_books = books_df.loc[
            ~books_df["ISBN"].astype(str).isin(rated_books)
        ].copy()

        candidate_books = candidate_books.drop_duplicates(
            subset="ISBN"
        )

        # --------------------------------------------------
        # Add user information
        # --------------------------------------------------
        candidate_books["User-ID"] = user_id
        candidate_books["Location"] = user_record["Location"]
        candidate_books["Age"] = user_record["Age"]

        # --------------------------------------------------
        # Keep only required model columns
        # --------------------------------------------------
        model_input = candidate_books[
            feature_columns
        ].copy()

        # --------------------------------------------------
        # Batch preprocessing
        # --------------------------------------------------
        X = preprocess_batch(
            model_input,
            encoders,
            scaler,
            feature_columns
        )

        # --------------------------------------------------
        # Batch prediction (FAST)
        # --------------------------------------------------
        predictions = model.predict(
            X,
            verbose=0
        ).flatten()

        predictions = np.clip(
            predictions,
            0,
            10
        )

        candidate_books["Predicted_Rating"] = predictions

        # --------------------------------------------------
        # Top recommendations
        # --------------------------------------------------
        recommendations = (
            candidate_books
            .sort_values(
                "Predicted_Rating",
                ascending=False
            )
            .head(top_n)
        )

        return recommendations[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Publisher",
                "Year-Of-Publication",
                "Predicted_Rating"
            ]
        ].reset_index(drop=True)

    except Exception as error:

        raise RuntimeError(
            f"Unable to generate recommendations: {error}"
        ) from error
    
def get_book_details(isbn: str, books_df: pd.DataFrame) -> dict[str, Any]:
    """Return selected metadata for the first book record matching an ISBN."""
    detail_columns = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication",
        "Image-URL-S",
        "Image-URL-M",
        "Image-URL-L",
    ]

    try:
        if not isinstance(books_df, pd.DataFrame):
            raise TypeError("books_df must be a pandas DataFrame.")
        if "ISBN" not in books_df.columns:
            raise ValueError("The books dataset must contain an ISBN column.")
        if isbn is None or pd.isna(isbn):
            return None

        # Compare string values to preserve ISBNs that include leading zeroes.
        matching_books = books_df.loc[
            books_df["ISBN"].astype(str) == str(isbn)
        ]
        if matching_books.empty:
            return None

        # Use only the first matching record when duplicate ISBN rows exist.
        book_record = matching_books.iloc[0]

        # Return a consistent response shape; unavailable columns and missing
        # values are represented as None for easy handling in Streamlit.
        return {
            column: (
                None
                if column not in books_df.columns or pd.isna(book_record.get(column))
                else book_record.get(column)
            )
            for column in detail_columns
        }

    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise RuntimeError(f"Unable to retrieve book details: {error}") from error
