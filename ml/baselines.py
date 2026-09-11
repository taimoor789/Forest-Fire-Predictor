"""
Stage 9 baselines (B0-B4), per docs/PREREGISTRATION.md's ablation ladder.
B2 -- get_danger_class, the live incumbent -- is the real bar the full
model must clear; B0/B1 are floor/sanity checks; B3/B4 isolate how much
signal sits in pure seasonality vs. pure ignition-history, independent of
FWI entirely.

B0-B2 are direct column transforms (no fitting). B3/B4 fit a plain
LogisticRegression on ONLY their named feature group, on the training rows
of whichever split regime is active, and score the test rows with it --
deliberately simple (not the tree ensemble Stage 10 uses for the real
model) since the point is to isolate how much a feature group carries on
its own, not to get the best possible model out of it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fire_risk import CanadianFireWeatherIndex
from ml import config

_fwi_calc = CanadianFireWeatherIndex()

# get_danger_class's own threshold order -- imported behavior, not a
# re-derived copy: same six classes, same ordinal rank.
_DANGER_CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"]


def score_b0(df, train_df):
    """Constant score (train base rate) for every row -- a classifier that
    carries no information. PR-AUC of a constant score equals the base
    rate; this exists as the explicit floor everything else is compared
    against, not just an implicit "0.5 is bad" assumption."""
    base_rate = train_df[f"label_w{config.PRIMARY_LABEL_WINDOW}"].mean()
    return np.full(len(df), base_rate)


def score_b1(df, train_df=None):
    """Raw FWI value as a ranking score -- current production physics,
    with no modelling layer on top."""
    return df["fwi"].values


def score_b2(df, train_df=None):
    """get_danger_class's own tier ordinal -- literally what's deployed
    today. Ties within a tier are real, not an artifact: production shows
    users the same six buckets, so any information loss from
    discretization is part of what this baseline is meant to capture."""
    ordinal_by_class = {name: i for i, name in enumerate(_DANGER_CLASS_ORDER)}
    classes = df["fwi"].apply(lambda fwi: _fwi_calc.get_danger_class(fwi)[0])
    return classes.map(ordinal_by_class).values.astype(float)


def _fit_and_score(df, train_df, feature_cols):
    target_col = f"label_w{config.PRIMARY_LABEL_WINDOW}"
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = df[feature_cols].values

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.SEED)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def score_b3(df, train_df):
    """Seasonality only: day_of_year (+ its sine/cosine, since day_of_year
    is circular -- Dec 31 and Jan 1 are adjacent, not 364 days apart) and
    month. Fit on train, scored on test -- never the reverse."""
    for frame in (df, train_df):
        if "day_of_year" not in frame.columns:
            frame["day_of_year"] = frame["date"].dt.dayofyear
        if "doy_sin" not in frame.columns:
            frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
            frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)
    return _fit_and_score(df, train_df, ["doy_sin", "doy_cos", "day_of_year"])


def score_b4(df, train_df):
    """historical_fire only (leak-free, Stage 2) -- pure ignition/fuel-
    proxy signal, independent of any weather."""
    return _fit_and_score(df, train_df, [
        "hist_fire_count_prior_20y_log1p", "hist_fire_any_prior", "years_since_last_fire",
    ])


BASELINES = {
    "B0_constant": score_b0,
    "B1_raw_fwi": score_b1,
    "B2_danger_class": score_b2,
    "B3_seasonality": score_b3,
    "B4_historical_fire": score_b4,
}
