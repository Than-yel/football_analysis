"""Pre-match feature-state helpers matched to the study notebook.

The original notebook's code for constructing historical features is retained
verbatim below; this module adds the serializable frozen inference wrapper.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd

__version__ = "0.1.0"
ROLLING_WINDOW = 5
LONG_WINDOW = 10

POST_MATCH_COLUMNS = {
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "MatchResult",
}
FORM_METRICS = [
    "points", "goals_for", "goals_against", "goal_diff", "shot_diff", "sot_diff",
    "corner_diff", "discipline", "win", "draw", "clean_sheet",
]
LONG_METRICS = ["points", "goals_for", "goals_against", "goal_diff"]
VENUE_METRICS = ["points", "goal_diff", "win"]
SEASON_METRICS = ["points", "goals_for", "goals_against", "goal_diff", "shot_diff", "sot_diff"]


def fresh_state() -> dict:
    return {
        "history": {},
        "venue_history": {},
        "season_stats": {},
        "last_played": {},
        "elo": {},
        "current_season": None,
    }


def prepare_season(state: dict, season: str) -> None:
    if state["current_season"] == season:
        return
    if state["current_season"] is not None:
        state["elo"] = {
            team: 1500.0 + 0.75 * (rating - 1500.0)
            for team, rating in state["elo"].items()
        }
    state["current_season"] = season


def mean_or_nan(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else np.nan


def team_snapshot(team: str, venue: str, timestamp: pd.Timestamp, season: str, state: dict) -> dict:
    history = state["history"].get(team, [])
    recent5 = history[-ROLLING_WINDOW:]
    recent10 = history[-LONG_WINDOW:]
    venue_history = state["venue_history"].get((team, venue), [])[-ROLLING_WINDOW:]
    season_key = (season, team)
    season_stats = state["season_stats"].get(season_key, {"matches": 0})
    season_matches = int(season_stats.get("matches", 0))

    previous_date = state["last_played"].get(team)
    rest_days = np.nan if previous_date is None else float(
        (timestamp.normalize() - previous_date.normalize()).days
    )
    if not np.isnan(rest_days):
        rest_days = float(np.clip(rest_days, 2, 30))

    snapshot = {
        "elo": float(state["elo"].get(team, 1500.0)),
        "history_matches": float(len(history)),
        "season_matches": float(season_matches),
        "rest_days": rest_days,
        "cold_start": float(len(history) < ROLLING_WINDOW),
    }
    for metric in FORM_METRICS:
        snapshot[f"form5_{metric}"] = mean_or_nan(recent5, metric)
    for metric in LONG_METRICS:
        snapshot[f"form10_{metric}"] = mean_or_nan(recent10, metric)
    for metric in VENUE_METRICS:
        snapshot[f"venue5_{metric}"] = mean_or_nan(venue_history, metric)
    for metric in SEASON_METRICS:
        total = float(season_stats.get(metric, 0.0))
        snapshot[f"season_{metric}"] = total / season_matches if season_matches else np.nan
    return snapshot


def fixture_snapshot(row: pd.Series, state: dict) -> dict:
    timestamp = pd.Timestamp(row["MatchDateTime"])
    home = team_snapshot(row["HomeTeam"], "H", timestamp, row["Season"], state)
    away = team_snapshot(row["AwayTeam"], "A", timestamp, row["Season"], state)
    feature_row = {
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "kickoff_hour": timestamp.hour + timestamp.minute / 60.0,
        "day_of_week": float(timestamp.dayofweek),
        "month": float(timestamp.month),
        "is_weekend": float(timestamp.dayofweek >= 5),
    }
    for key in home:
        feature_row[f"home_{key}"] = home[key]
        feature_row[f"away_{key}"] = away[key]
        feature_row[f"diff_{key}"] = home[key] - away[key]
    return feature_row


def team_match_record(row: pd.Series, home_side: bool) -> dict:
    if home_side:
        goals_for, goals_against = row["FTHG"], row["FTAG"]
        shots_for, shots_against = row["HS"], row["AS"]
        sot_for, sot_against = row["HST"], row["AST"]
        corners_for, corners_against = row["HC"], row["AC"]
        yellow, red = row["HY"], row["HR"]
    else:
        goals_for, goals_against = row["FTAG"], row["FTHG"]
        shots_for, shots_against = row["AS"], row["HS"]
        sot_for, sot_against = row["AST"], row["HST"]
        corners_for, corners_against = row["AC"], row["HC"]
        yellow, red = row["AY"], row["AR"]

    points = 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)
    return {
        "points": float(points),
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "goal_diff": float(goals_for - goals_against),
        "shot_diff": float(shots_for - shots_against),
        "sot_diff": float(sot_for - sot_against),
        "corner_diff": float(corners_for - corners_against),
        "discipline": float(yellow + 3 * red),
        "win": float(goals_for > goals_against),
        "draw": float(goals_for == goals_against),
        "clean_sheet": float(goals_against == 0),
    }


def update_team_state(team: str, venue: str, season: str, timestamp: pd.Timestamp, record: dict, state: dict) -> None:
    state["history"].setdefault(team, []).append(record)
    state["venue_history"].setdefault((team, venue), []).append(record)
    season_key = (season, team)
    accumulator = state["season_stats"].setdefault(
        season_key, {"matches": 0, **{metric: 0.0 for metric in SEASON_METRICS}}
    )
    accumulator["matches"] += 1
    for metric in SEASON_METRICS:
        accumulator[metric] += record[metric]
    state["last_played"][team] = timestamp


def update_after_match(row: pd.Series, state: dict) -> None:
    timestamp = pd.Timestamp(row["MatchDateTime"])
    home_team, away_team = row["HomeTeam"], row["AwayTeam"]
    home_record = team_match_record(row, home_side=True)
    away_record = team_match_record(row, home_side=False)
    update_team_state(home_team, "H", row["Season"], timestamp, home_record, state)
    update_team_state(away_team, "A", row["Season"], timestamp, away_record, state)

    home_rating = float(state["elo"].get(home_team, 1500.0))
    away_rating = float(state["elo"].get(away_team, 1500.0))
    expected_home = 1.0 / (1.0 + 10.0 ** (-(home_rating + 65.0 - away_rating) / 400.0))
    actual_home = 1.0 if row["MatchResult"] == 0 else (0.5 if row["MatchResult"] == 1 else 0.0)
    adjustment = 20.0 * (actual_home - expected_home)
    state["elo"][home_team] = home_rating + adjustment
    state["elo"][away_team] = away_rating - adjustment


def assert_no_team_repeats_at_kickoff(batch: pd.DataFrame) -> None:
    teams = pd.concat([batch["HomeTeam"], batch["AwayTeam"]], ignore_index=True)
    assert not teams.duplicated().any(), "A team appears twice at the same kickoff timestamp."


def build_historical_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    work = matches.sort_values(["MatchDateTime", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    state = fresh_state()
    feature_rows, targets, metadata_rows = [], [], []

    for timestamp, batch in work.groupby("MatchDateTime", sort=True):
        assert_no_team_repeats_at_kickoff(batch)
        season = str(batch["Season"].iloc[0])
        prepare_season(state, season)

        # Snapshot the whole kickoff group before any result at that time is applied.
        for _, row in batch.iterrows():
            feature_rows.append(fixture_snapshot(row, state))
            targets.append(int(row["MatchResult"]))
            metadata_rows.append(
                {
                    "MatchDateTime": pd.Timestamp(timestamp),
                    "Season": row["Season"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                }
            )
        for _, row in batch.iterrows():
            update_after_match(row, state)

    return pd.DataFrame(feature_rows), np.asarray(targets), pd.DataFrame(metadata_rows), state


def build_upcoming_features(fixtures: pd.DataFrame, historical_state: dict) -> pd.DataFrame:
    frozen_state = copy.deepcopy(historical_state)
    rows = []
    ordered = fixtures.sort_values(["MatchDateTime", "FixtureID"])
    for timestamp, batch in ordered.groupby("MatchDateTime", sort=True):
        assert_no_team_repeats_at_kickoff(batch)
        prepare_season(frozen_state, str(batch["Season"].iloc[0]))
        for _, row in batch.iterrows():
            rows.append({"FixtureID": row["FixtureID"], **fixture_snapshot(row, frozen_state)})
        # Deliberately no update: all fixtures use the same completed-match cutoff.
    transformed = pd.DataFrame(rows).set_index("FixtureID")
    requested_ids = fixtures["FixtureID"].tolist()
    transformed = transformed.loc[requested_ids]
    expected = fixtures.set_index("FixtureID").loc[requested_ids, ["HomeTeam", "AwayTeam"]]
    assert transformed[["HomeTeam", "AwayTeam"]].equals(expected)
    return transformed


class FrozenEPLFeatureBuilder:
    """Rebuild strictly pre-kickoff features from frozen completed-match history."""

    def __init__(self, historical_state: dict, feature_columns: list[str],
                 rolling_window: int = 5, long_window: int = 10):
        self.historical_state = copy.deepcopy(historical_state)
        self.feature_columns = list(feature_columns)
        self.rolling_window = int(rolling_window)
        self.long_window = int(long_window)
        if self.rolling_window != ROLLING_WINDOW or self.long_window != LONG_WINDOW:
            raise ValueError("This version supports the fixed 5/10-match windows used in the study.")

    def transform(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        required = {"FixtureID", "MatchDateTime", "HomeTeam", "AwayTeam", "Season"}
        absent = required.difference(fixtures.columns)
        if absent:
            raise ValueError(f"Missing fixture columns: {sorted(absent)}")
        frame = build_upcoming_features(fixtures, self.historical_state)
        absent_features = set(self.feature_columns).difference(frame.columns)
        if absent_features:
            raise ValueError(f"Missing engineered features: {sorted(absent_features)}")
        return frame.loc[:, self.feature_columns]
