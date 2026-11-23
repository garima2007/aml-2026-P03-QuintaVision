"""Shared modelling utilities for QuintaVision P03.

This module centralises prediction-time feature policy and preprocessing
used by the modelling notebooks.

Raw-data cleaning remains the responsibility of Notebook 03.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "is_canceled"

# Direct outcome leakage is already removed by Notebook 03.
# The remaining three are excluded because they are not considered
# safely available at the original booking-time decision point.
BOOKING_TIME_EXCLUSIONS = [
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]


def prepare_model_data(df):
    """Create the canonical booking-time modelling table.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data produced by Notebook 03.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered, booking-time-safe modelling table.
    """
    data = df.copy()

    # Defensive guard against direct outcome leakage in case a caller
    # supplies something other than Notebook 03's cleaned output.
    direct_leakage = [
        "reservation_status",
        "reservation_status_date",
    ]

    drop_now = [
        c for c in direct_leakage + BOOKING_TIME_EXCLUSIONS
        if c in data.columns
    ]

    if drop_now:
        data = data.drop(columns=drop_now)

    required = [
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "total_of_special_requests",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Required columns missing from cleaned data: "
            + ", ".join(missing)
        )

    data["total_nights"] = (
        data["stays_in_weekend_nights"]
        + data["stays_in_week_nights"]
    )

    data["total_guests"] = (
        data["adults"]
        + data["children"]
        + data["babies"]
    )

    data["total_previous_bookings"] = (
        data["previous_cancellations"]
        + data["previous_bookings_not_canceled"]
    )

    data["has_special_request"] = (
        data["total_of_special_requests"] > 0
    ).astype(int)

    redundant = [
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
    ]

    data = data.drop(
        columns=[c for c in redundant if c in data.columns]
    )

    return data


class RareCountryGrouper(BaseEstimator, TransformerMixin):
    """Group country levels below a fitted frequency threshold into Other."""

    def __init__(self, column="country", min_frequency=0.01):
        self.column = column
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        # ColumnTransformer supplies a DataFrame for the categorical block
        # in the current project workflow.
        if hasattr(X, "columns"):
            X_df.columns = X.columns

        if self.column in X_df.columns:
            freq = X_df[self.column].value_counts(
                normalize=True,
                dropna=False,
            )
            self.common_countries_ = set(
                freq[freq >= self.min_frequency].index
            )
        else:
            self.common_countries_ = set()

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        if hasattr(X, "columns"):
            X_df.columns = X.columns

        if self.column in X_df.columns:
            X_df[self.column] = X_df[self.column].where(
                X_df[self.column].isin(self.common_countries_),
                "Other",
            )

        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)

        return np.asarray(input_features, dtype=object)


def build_preprocessors(X_train):
    """Build the shared standard and boosting preprocessors."""

    numerical_features = (
        X_train
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        (
            "rare_country",
            RareCountryGrouper(min_frequency=0.01),
        ),
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numerical_features),
        ("cat", cat_pipe, categorical_features),
    ])

    boost_cat_pipe = Pipeline([
        (
            "rare_country",
            RareCountryGrouper(min_frequency=0.01),
        ),
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    boost_preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_features,
        ),
        (
            "cat",
            boost_cat_pipe,
            categorical_features,
        ),
    ])

    return (
        preprocessor,
        boost_preprocessor,
        numerical_features,
        categorical_features,
    )


def stratified_train_test_split(X, y):
    """Return the project's canonical reproducible 80/20 split."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def make_cv():
    """Return the project's canonical shuffled stratified CV object."""
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_random_forest():
    """Return the canonical Random Forest used in final modelling."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
