"""Release-ready GridLock 2.0 forecasting pipeline."""

from __future__ import annotations

import json
import lzma
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional dependency in local runs
    kagglehub = None


NUM_BASIS_DAYS = 12
RIDGE_ALPHA = 10.0
RESIDUAL_ALPHA = 0.30
PROXY_MORNING_END = 120
PROXY_EVENING_START = 840
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "disp": False}
ALL_SLOTS = np.arange(0, 24 * 60, 15, dtype=int)
VISIBLE_SLOTS = list(range(0, PROXY_MORNING_END + 15, 15))
HIDDEN_DAY_SLOTS = list(range(PROXY_MORNING_END + 15, PROXY_EVENING_START, 15))
NON_OVERLAP_SLOTS = VISIBLE_SLOTS + list(range(PROXY_EVENING_START, 1440, 15))
VALIDATION_DAYS = [42, 35, 28]


def detect_release_root() -> Path:
    """Resolve the release root from script or notebook execution contexts."""
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)

    current_dir = Path.cwd().resolve()
    candidates.extend([current_dir, current_dir.parent, current_dir.parent.parent])
    for candidate in candidates:
        if (candidate / "gridlock_release_pipeline.py").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the GridLock 2.0 release root. Expected gridlock_release_pipeline.py and data/."
    )


RELEASE_ROOT = detect_release_root()

DATA_DIR = RELEASE_ROOT / "data"
ARTIFACTS_DIR = RELEASE_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "gridlock_release_model.pkl"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission.csv"
VALIDATION_PATH = ARTIFACTS_DIR / "validation_summary.json"
RUN_LOG_PATH = ARTIFACTS_DIR / "run_log.txt"
CONDITION_REPORT_PATH = ARTIFACTS_DIR / "condition_matrix_report.txt"
LOCAL_SCORE_PATH = DATA_DIR / "evaluation_ground_truth.csv"
HISTORICAL_XZ_PATH = DATA_DIR / "historical_training.csv.xz"
HISTORICAL_CSV_PATH = DATA_DIR / "historical_training.csv"


@dataclass
class CalibratorArtifact:
    """Serialized ridge calibrator state."""

    alpha: float
    intercept: float
    coefficients: list[float]
    feature_columns: list[str]


@dataclass
class ModelArtifact:
    """Serialized inference artifact for the isolated submission bundle."""

    basis_days: list[int]
    global_weights: dict[int, np.ndarray]
    sector_weights: dict[str, dict[int, np.ndarray]]
    subsector_weights: dict[str, dict[int, np.ndarray]]
    lower_boundary: float
    upper_boundary: float
    geo_residual: dict[str, float]
    calibrator: CalibratorArtifact
    metrics: dict[str, float | int | str]


@dataclass
class InputAvailability:
    """Resolved dataset inputs together with missing-path diagnostics."""

    historical_path: Path | None
    train_path: Path | None
    test_path: Path | None
    missing_labels: list[str]


def timestamp_to_minutes(timestamp: str) -> int:
    """Convert an HH:MM timestamp into minutes since midnight."""
    hour, minute = map(int, str(timestamp).split(":"))
    return hour * 60 + minute


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add time and geohash helper columns."""
    prepared = frame.copy().reset_index(drop=True)
    prepared["time_min"] = prepared["timestamp"].apply(timestamp_to_minutes)
    prepared["geo_4"] = prepared["geohash"].astype(str).str.slice(0, 4)
    prepared["geo_5"] = prepared["geohash"].astype(str).str.slice(0, 5)
    return prepared


def _candidate_data_roots() -> list[Path]:
    """Return local candidate roots that may contain the datasets."""
    roots = [RELEASE_ROOT]
    if kagglehub is not None:
        try:
            roots.append(Path(kagglehub.dataset_download("kweklydia5/grabtrafficdata")))
        except Exception:
            pass
    return roots


def _resolve_existing_path(relative_candidates: list[str]) -> Path | None:
    """Resolve the first existing path from the known dataset roots."""
    for root in _candidate_data_roots():
        for candidate in relative_candidates:
            path = root / candidate
            if path.exists():
                return path
    return None


def resolve_input_paths() -> InputAvailability:
    """Resolve all required dataset paths without raising."""
    historical_path = _resolve_existing_path(
        [
            "data/historical_training.csv.xz",
            "data/historical_training.csv",
            "training.csv",
            "dataset/training.csv",
        ]
    )
    train_path = _resolve_existing_path(
        [
            "data/competition_train.csv",
            "dataset/train.csv",
            "train.csv",
        ]
    )
    test_path = _resolve_existing_path(
        [
            "data/competition_test.csv",
            "dataset/test.csv",
            "test.csv",
        ]
    )
    missing_labels: list[str] = []
    if historical_path is None:
        missing_labels.append("data/historical_training.csv(.xz)")
    if train_path is None:
        missing_labels.append("data/competition_train.csv")
    if test_path is None:
        missing_labels.append("data/competition_test.csv")
    return InputAvailability(
        historical_path=historical_path,
        train_path=train_path,
        test_path=test_path,
        missing_labels=missing_labels,
    )


def ensure_historical_csv_from_xz(source_path: Path, target_path: Path) -> Path:
    """Extract a compressed historical file into plain CSV form when needed."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(source_path, "rb") as src, target_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return target_path


def load_historical_frame(path: Path) -> pd.DataFrame:
    """Load the historical dataset with a fallback path for compressed input."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if path.suffix != ".xz":
            raise RuntimeError(f"Unable to load historical dataset from {path}") from exc
        extracted_path = ensure_historical_csv_from_xz(path, HISTORICAL_CSV_PATH)
        try:
            return pd.read_csv(extracted_path)
        except Exception as extracted_exc:
            raise RuntimeError(
                "Unable to load the compressed historical dataset or its extracted CSV fallback."
            ) from extracted_exc


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the historical, competition-train, and competition-test files."""
    availability = resolve_input_paths()
    if availability.missing_labels:
        missing_text = ", ".join(availability.missing_labels)
        raise FileNotFoundError(f"Missing required dataset files: {missing_text}")

    historical = load_historical_frame(availability.historical_path)
    if "geohash6" in historical.columns:
        historical = historical.rename(columns={"geohash6": "geohash"})

    try:
        train = pd.read_csv(availability.train_path)
        test = pd.read_csv(availability.test_path).sort_values("Index").reset_index(drop=True)
    except Exception as exc:
        raise RuntimeError("Unable to load the competition train/test files.") from exc
    return historical, train, test


def emit(message: str, log_lines: list[str]) -> None:
    """Print a message and append it to the run log buffer."""
    print(message)
    log_lines.append(message)


def prepare_competition_slice(train: pd.DataFrame) -> pd.DataFrame:
    """Select the observed Day 49 competition slice used for calibrator fitting."""
    observed = prepare_frame(train[train["day"] == 49].copy())
    return observed[observed["time_min"] <= PROXY_MORNING_END].reset_index(drop=True)


def prepare_non_overlap_proxy(historical: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Select Day 49 labels that do not overlap the official test horizon."""
    day_49 = prepare_frame(historical[historical["day"] == 49].copy())
    test_slots = set(test["time_min"].unique().tolist())
    proxy = day_49[~day_49["time_min"].isin(test_slots)].copy()
    return proxy.sort_values(["time_min", "geohash"]).reset_index(drop=True)


def build_day_lookup_cache(historical: pd.DataFrame) -> dict[int, pd.Series]:
    """Build day-level demand lookup tables indexed by geohash and time."""
    cache: dict[int, pd.Series] = {}
    for day in sorted(historical["day"].unique()):
        day_frame = historical[historical["day"] == day]
        cache[int(day)] = day_frame.set_index(["geohash", "time_min"])["demand"]
    return cache


def select_basis_days(
    frame: pd.DataFrame,
    day_lookup_cache: dict[int, pd.Series],
    target: np.ndarray,
    excluded_days: set[int] | None = None,
    top_k: int = NUM_BASIS_DAYS,
) -> list[int]:
    """Select basis days by correlation against a proxy target."""
    excluded_days = excluded_days or set()
    frame_index = frame.set_index(["geohash", "time_min"]).index
    target_std = float(target.std())
    correlations: list[tuple[int, float]] = []

    for day, lookup in day_lookup_cache.items():
        if day in excluded_days:
            continue
        candidate = frame_index.map(lookup).fillna(0.0).to_numpy(dtype=np.float64)
        candidate_std = float(candidate.std())
        if target_std == 0.0 or candidate_std == 0.0:
            continue
        corr = float(np.corrcoef(target, candidate)[0, 1])
        correlations.append((day, corr))

    correlations.sort(key=lambda item: item[1], reverse=True)
    return [day for day, _ in correlations[:top_k]]


def build_components(
    historical: pd.DataFrame,
    frame: pd.DataFrame,
    basis_days: list[int],
    day_lookup_cache: dict[int, pd.Series],
    excluded_global_days: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the global baseline and per-basis-day component matrices."""
    frame_index = frame.set_index(["geohash", "time_min"]).index
    excluded_global_days = excluded_global_days or set()
    global_average = historical[~historical["day"].isin(excluded_global_days)].groupby(
        ["geohash", "time_min"]
    )["demand"].mean()
    component_global = frame_index.map(global_average).fillna(0.0).to_numpy(dtype=np.float64)

    component_days = []
    for day in basis_days:
        component_days.append(frame_index.map(day_lookup_cache[day]).fillna(0.0).to_numpy(dtype=np.float64))
    return component_global, np.column_stack(component_days)


def fit_group_weights(group_df: pd.DataFrame, basis_days: list[int]) -> np.ndarray:
    """Fit simplex-constrained routing weights for a single group."""
    y_values = group_df["y_true"].to_numpy(dtype=np.float64)
    global_values = group_df["pGlobal"].to_numpy(dtype=np.float64)
    basis_values = [group_df[f"p{day}"].to_numpy(dtype=np.float64) for day in basis_days]
    num_params = len(basis_days) + 1
    init = np.full(num_params, 1.0 / num_params, dtype=np.float64)

    if len(y_values) < 5 or float(np.var(y_values)) == 0.0:
        return init

    def objective(weights: np.ndarray) -> float:
        prediction = (weights[0] * global_values) + sum(
            weights[index + 1] * basis_values[index] for index in range(len(basis_days))
        )
        return -r2_score(y_values, prediction)

    constraints = ({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},)
    bounds = [(0.0, 1.0)] * num_params
    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=OPTIMIZER_OPTIONS,
    )
    return result.x if result.success else init


def fit_hierarchical_weights(
    frame: pd.DataFrame,
    proxy_target: np.ndarray,
    component_global: np.ndarray,
    component_days: np.ndarray,
    basis_days: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    """Fit global, geo4, and geo5 routing weights from the proxy target."""
    proxy_df = pd.DataFrame(
        {
            "geo_4": frame["geo_4"].values,
            "geo_5": frame["geo_5"].values,
            "time_min": frame["time_min"].values,
            "y_true": proxy_target,
            "pGlobal": component_global,
        }
    )
    for index, day in enumerate(basis_days):
        proxy_df[f"p{day}"] = component_days[:, index]

    global_weights: dict[int, np.ndarray] = {}
    for slot in sorted(proxy_df["time_min"].unique()):
        global_weights[int(slot)] = fit_group_weights(proxy_df[proxy_df["time_min"] == slot], basis_days)

    sector_weights: dict[str, dict[int, np.ndarray]] = {}
    for (geo_4, slot), group_df in proxy_df.groupby(["geo_4", "time_min"]):
        sector_weights.setdefault(geo_4, {})[int(slot)] = fit_group_weights(group_df, basis_days)

    subsector_weights: dict[str, dict[int, np.ndarray]] = {}
    for (geo_5, slot), group_df in proxy_df.groupby(["geo_5", "time_min"]):
        geo_4 = geo_5[:4]
        if len(group_df) < 14 or float(np.var(group_df["y_true"].to_numpy(dtype=np.float64))) == 0.0:
            weights = sector_weights.get(geo_4, {}).get(int(slot), global_weights[int(slot)])
        else:
            weights = fit_group_weights(group_df, basis_days)
        subsector_weights.setdefault(geo_5, {})[int(slot)] = weights

    return global_weights, sector_weights, subsector_weights


def interpolate_weight_tables(
    global_weights: dict[int, np.ndarray],
    sector_weights: dict[str, dict[int, np.ndarray]],
    subsector_weights: dict[str, dict[int, np.ndarray]],
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    """Interpolate observed proxy-slot weights onto the full 96-slot grid."""

    def interpolate_group(slot_map: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        observed_slots = np.array(sorted(slot_map.keys()), dtype=int)
        if observed_slots.size == 0:
            return {}

        weights_matrix = np.vstack([slot_map[int(slot)] for slot in observed_slots])
        num_params = weights_matrix.shape[1]
        full_weights = np.zeros((len(ALL_SLOTS), num_params), dtype=np.float64)
        for column_index in range(num_params):
            full_weights[:, column_index] = np.interp(
                ALL_SLOTS,
                observed_slots,
                weights_matrix[:, column_index],
            )
        full_weights = np.clip(full_weights, 0.0, None)
        row_sums = full_weights.sum(axis=1, keepdims=True)
        full_weights = np.divide(
            full_weights,
            row_sums,
            out=np.full_like(full_weights, 1.0 / num_params),
            where=row_sums > 0.0,
        )
        return {int(slot): full_weights[index] for index, slot in enumerate(ALL_SLOTS)}

    interpolated_global = interpolate_group(global_weights)
    interpolated_sector = {geo_key: interpolate_group(slot_map) for geo_key, slot_map in sector_weights.items()}
    interpolated_subsector = {
        geo_key: interpolate_group(slot_map) for geo_key, slot_map in subsector_weights.items()
    }
    return interpolated_global, interpolated_sector, interpolated_subsector


def predict_matrix(
    frame: pd.DataFrame,
    basis_days: list[int],
    component_global: np.ndarray,
    component_days: np.ndarray,
    global_weights: dict[int, np.ndarray],
    sector_weights: dict[str, dict[int, np.ndarray]],
    subsector_weights: dict[str, dict[int, np.ndarray]],
) -> np.ndarray:
    """Reconstruct predictions from the fitted hierarchical routing weights."""
    predictions = np.zeros(len(frame), dtype=np.float64)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        slot = int(row.time_min)
        weights = subsector_weights.get(row.geo_5, {}).get(slot)
        if weights is None:
            weights = sector_weights.get(row.geo_4, {}).get(slot)
        if weights is None:
            weights = global_weights[slot]
        predictions[row_index] = (weights[0] * component_global[row_index]) + sum(
            weights[day_index + 1] * component_days[row_index, day_index]
            for day_index in range(len(basis_days))
        )
    return predictions


def predict_matrix_with_slot_fallback(
    frame: pd.DataFrame,
    basis_days: list[int],
    component_global: np.ndarray,
    component_days: np.ndarray,
    global_weights: dict[int, np.ndarray],
    sector_weights: dict[str, dict[int, np.ndarray]],
    subsector_weights: dict[str, dict[int, np.ndarray]],
) -> np.ndarray:
    """Predict a frame even when some slots fall outside the fitted slot grid."""
    available_slots = np.array(sorted(global_weights.keys()), dtype=int)
    predictions = np.zeros(len(frame), dtype=np.float64)

    for row_index, row in enumerate(frame.itertuples(index=False)):
        slot = int(row.time_min)
        weights = subsector_weights.get(row.geo_5, {}).get(slot)
        if weights is None:
            weights = sector_weights.get(row.geo_4, {}).get(slot)
        if weights is None and slot in global_weights:
            weights = global_weights[slot]
        if weights is None:
            nearest_slot = int(available_slots[int(np.argmin(np.abs(available_slots - slot)))])
            weights = global_weights[nearest_slot]

        predictions[row_index] = (weights[0] * component_global[row_index]) + sum(
            weights[day_index + 1] * component_days[row_index, day_index]
            for day_index in range(len(basis_days))
        )
    return predictions


def infer_boundaries(test: pd.DataFrame, predictions: np.ndarray) -> tuple[float, float]:
    """Infer clipping thresholds from the test-side road-type distribution."""
    road_type = test["RoadType"].dropna()
    residential_share = float((road_type == "Residential").mean())
    street_share = float((road_type == "Street").mean())
    lower = float(np.percentile(predictions, residential_share * 100.0))
    upper = float(np.percentile(predictions, (residential_share + street_share) * 100.0))
    return lower, upper


def apply_feature_boundaries(
    predictions: np.ndarray,
    test: pd.DataFrame,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Apply category-consistent clipping using non-target test-side features."""
    bounded = predictions.copy()
    for row_index, row in enumerate(test.itertuples(index=False)):
        road_type = row.RoadType
        large_vehicles = row.LargeVehicles
        lanes = pd.to_numeric(row.NumberofLanes, errors="coerce")

        if pd.notna(road_type) and str(road_type) not in {"_missing_", "nan"}:
            if road_type == "Residential":
                bounded[row_index] = np.clip(bounded[row_index], 0.0, lower)
            elif road_type == "Street":
                bounded[row_index] = np.clip(bounded[row_index], lower, upper)
            elif road_type == "Highway":
                bounded[row_index] = np.clip(bounded[row_index], upper, 1.0)
        else:
            if large_vehicles == "Not Allowed" or lanes == 1:
                bounded[row_index] = np.clip(bounded[row_index], 0.0, upper)
            elif lanes in [4, 5]:
                bounded[row_index] = np.clip(bounded[row_index], upper, 1.0)

        if large_vehicles == "Not Allowed" or lanes == 1:
            bounded[row_index] = np.clip(bounded[row_index], 0.0, upper)
        elif lanes in [4, 5]:
            bounded[row_index] = np.clip(bounded[row_index], upper, 1.0)
    return bounded


def _build_calibrator_features(frame: pd.DataFrame, base_predictions: np.ndarray) -> pd.DataFrame:
    """Build the calibrator feature matrix."""
    slot_index = frame["time_min"].to_numpy(dtype=np.float64) / 15.0
    features = pd.DataFrame(
        {
            "base": base_predictions,
            "time_min": frame["time_min"].to_numpy(dtype=np.float64),
            "time_sin": np.sin(2.0 * np.pi * slot_index / 96.0),
            "time_cos": np.cos(2.0 * np.pi * slot_index / 96.0),
            "geo_4": frame["geo_4"].astype(str),
        }
    )
    return pd.get_dummies(features, columns=["geo_4"], drop_first=False)


def fit_calibrator(frame: pd.DataFrame, base_predictions: np.ndarray, targets: np.ndarray) -> CalibratorArtifact:
    """Fit and serialize the ridge calibrator."""
    features = _build_calibrator_features(frame, base_predictions)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(features, targets)
    return CalibratorArtifact(
        alpha=RIDGE_ALPHA,
        intercept=float(model.intercept_),
        coefficients=model.coef_.astype(np.float64).tolist(),
        feature_columns=list(features.columns),
    )


def apply_calibrator(frame: pd.DataFrame, base_predictions: np.ndarray, artifact: CalibratorArtifact) -> np.ndarray:
    """Apply the serialized ridge calibrator."""
    features = _build_calibrator_features(frame, base_predictions)
    features = features.reindex(columns=artifact.feature_columns, fill_value=0.0)
    matrix = features.to_numpy(dtype=np.float64)
    return matrix @ np.asarray(artifact.coefficients, dtype=np.float64) + artifact.intercept


def compute_geo_residual(historical: pd.DataFrame) -> dict[str, float]:
    """Compute a per-geohash non-overlap residual between Day 49 and Day 42."""
    geo42 = (
        historical[(historical["day"] == 42) & (historical["time_min"].isin(NON_OVERLAP_SLOTS))]
        .groupby("geohash")["demand"]
        .mean()
    )
    geo49 = (
        historical[(historical["day"] == 49) & (historical["time_min"].isin(NON_OVERLAP_SLOTS))]
        .groupby("geohash")["demand"]
        .mean()
    )
    residual = geo49.subtract(geo42, fill_value=0.0).fillna(0.0)
    return {str(key): float(value) for key, value in residual.items()}


def apply_geo_residual(predictions: np.ndarray, frame: pd.DataFrame, geo_residual: dict[str, float]) -> np.ndarray:
    """Apply a bounded per-geohash additive correction."""
    corrected = predictions.astype(np.float64, copy=True)
    geohashes = frame["geohash"].astype(str).to_numpy()
    for row_index, geohash in enumerate(geohashes):
        corrected[row_index] = np.clip(corrected[row_index] + RESIDUAL_ALPHA * geo_residual.get(geohash, 0.0), 0.0, 1.0)
    return corrected


def train_model_artifact(
    historical: pd.DataFrame,
    competition_train: pd.DataFrame,
    test: pd.DataFrame,
) -> ModelArtifact:
    """Train the submission artifact from the available data files."""
    historical = historical.copy()
    historical["time_min"] = historical["timestamp"].apply(timestamp_to_minutes)
    test = prepare_frame(test)
    competition_slice = prepare_competition_slice(competition_train)
    proxy_frame = prepare_non_overlap_proxy(historical, test)

    day_lookup_cache = build_day_lookup_cache(historical)
    proxy_target = proxy_frame["demand"].to_numpy(dtype=np.float64)
    basis_days = select_basis_days(proxy_frame, day_lookup_cache, proxy_target, excluded_days={49})

    component_global, component_days = build_components(
        historical,
        proxy_frame,
        basis_days,
        day_lookup_cache,
        excluded_global_days={49},
    )
    global_weights, sector_weights, subsector_weights = fit_hierarchical_weights(
        proxy_frame,
        proxy_target,
        component_global,
        component_days,
        basis_days,
    )
    global_weights, sector_weights, subsector_weights = interpolate_weight_tables(
        global_weights,
        sector_weights,
        subsector_weights,
    )

    test_global, test_days = build_components(
        historical,
        test,
        basis_days,
        day_lookup_cache,
        excluded_global_days={49},
    )
    matrix_predictions = predict_matrix(
        test,
        basis_days,
        test_global,
        test_days,
        global_weights,
        sector_weights,
        subsector_weights,
    )
    lower_boundary, upper_boundary = infer_boundaries(test, matrix_predictions)

    slice_global, slice_days = build_components(
        historical,
        competition_slice,
        basis_days,
        day_lookup_cache,
        excluded_global_days={49},
    )
    slice_base = predict_matrix_with_slot_fallback(
        competition_slice,
        basis_days,
        slice_global,
        slice_days,
        global_weights,
        sector_weights,
        subsector_weights,
    )
    calibrator = fit_calibrator(
        competition_slice,
        slice_base,
        competition_slice["demand"].to_numpy(dtype=np.float64),
    )
    geo_residual = compute_geo_residual(historical)

    metrics: dict[str, float | int | str] = {
        "basis_days": len(basis_days),
        "proxy_rows": len(proxy_frame),
        "competition_calibration_rows": len(competition_slice),
        "residual_alpha": RESIDUAL_ALPHA,
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
    }
    return ModelArtifact(
        basis_days=basis_days,
        global_weights=global_weights,
        sector_weights=sector_weights,
        subsector_weights=subsector_weights,
        lower_boundary=lower_boundary,
        upper_boundary=upper_boundary,
        geo_residual=geo_residual,
        calibrator=calibrator,
        metrics=metrics,
    )


def save_model_artifact(artifact: ModelArtifact) -> None:
    """Persist the trained submission artifact."""
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(asdict(artifact), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_artifact() -> ModelArtifact | None:
    """Load a previously trained submission artifact if it exists."""
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, ModelArtifact):
        return payload
    if isinstance(payload, dict):
        calibrator_payload = payload["calibrator"]
        calibrator = CalibratorArtifact(
            alpha=float(calibrator_payload["alpha"]),
            intercept=float(calibrator_payload["intercept"]),
            coefficients=[float(value) for value in calibrator_payload["coefficients"]],
            feature_columns=[str(value) for value in calibrator_payload["feature_columns"]],
        )
        global_weights = {
            int(slot): np.asarray(weights, dtype=np.float64)
            for slot, weights in payload["global_weights"].items()
        }
        sector_weights = {
            str(geo_key): {
                int(slot): np.asarray(weights, dtype=np.float64)
                for slot, weights in slot_map.items()
            }
            for geo_key, slot_map in payload["sector_weights"].items()
        }
        subsector_weights = {
            str(geo_key): {
                int(slot): np.asarray(weights, dtype=np.float64)
                for slot, weights in slot_map.items()
            }
            for geo_key, slot_map in payload["subsector_weights"].items()
        }
        return ModelArtifact(
            basis_days=[int(day) for day in payload["basis_days"]],
            global_weights=global_weights,
            sector_weights=sector_weights,
            subsector_weights=subsector_weights,
            lower_boundary=float(payload["lower_boundary"]),
            upper_boundary=float(payload["upper_boundary"]),
            geo_residual={str(key): float(value) for key, value in payload["geo_residual"].items()},
            calibrator=calibrator,
            metrics=dict(payload["metrics"]),
        )
    raise TypeError(f"Unsupported artifact payload type: {type(payload)!r}")


def run_inference(
    artifact: ModelArtifact,
    historical: pd.DataFrame,
    raw_test: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference from a trained artifact and return the submission frame."""
    historical = historical.copy()
    historical["time_min"] = historical["timestamp"].apply(timestamp_to_minutes)
    test = prepare_frame(raw_test)
    day_lookup_cache = build_day_lookup_cache(historical)

    test_global, test_days = build_components(
        historical,
        test,
        artifact.basis_days,
        day_lookup_cache,
        excluded_global_days={49},
    )
    matrix_predictions = predict_matrix(
        test,
        artifact.basis_days,
        test_global,
        test_days,
        artifact.global_weights,
        artifact.sector_weights,
        artifact.subsector_weights,
    )
    bounded_predictions = apply_feature_boundaries(
        matrix_predictions,
        test,
        artifact.lower_boundary,
        artifact.upper_boundary,
    )
    final_predictions = apply_calibrator(test, bounded_predictions, artifact.calibrator)
    final_predictions = apply_geo_residual(final_predictions, test, artifact.geo_residual)

    submission = raw_test[["Index"]].copy()
    submission["demand"] = final_predictions
    return submission


def evaluate_proxy_folds(
    historical: pd.DataFrame,
    competition_train: pd.DataFrame,
) -> dict[str, object]:
    """Run historical proxy folds to estimate generalization without hidden test targets."""
    historical = historical.copy()
    historical["time_min"] = historical["timestamp"].apply(timestamp_to_minutes)
    competition_train = prepare_frame(competition_train)
    day_lookup_cache = build_day_lookup_cache(historical)
    fold_results: list[dict[str, object]] = []

    for target_day in VALIDATION_DAYS:
        held_out = prepare_frame(historical[historical["day"] == target_day].copy())
        proxy_frame = held_out[held_out["time_min"].isin(NON_OVERLAP_SLOTS)].copy().reset_index(drop=True)
        target_frame = held_out[held_out["time_min"].isin(HIDDEN_DAY_SLOTS)].copy().reset_index(drop=True)
        calibration_frame = held_out[held_out["time_min"].isin(VISIBLE_SLOTS)].copy().reset_index(drop=True)

        proxy_target = proxy_frame["demand"].to_numpy(dtype=np.float64)
        basis_days = select_basis_days(proxy_frame, day_lookup_cache, proxy_target, excluded_days={49, target_day})
        component_global, component_days = build_components(
            historical,
            proxy_frame,
            basis_days,
            day_lookup_cache,
            excluded_global_days={49, target_day},
        )
        global_weights, sector_weights, subsector_weights = fit_hierarchical_weights(
            proxy_frame,
            proxy_target,
            component_global,
            component_days,
            basis_days,
        )
        global_weights, sector_weights, subsector_weights = interpolate_weight_tables(
            global_weights,
            sector_weights,
            subsector_weights,
        )

        target_global, target_days = build_components(
            historical,
            target_frame,
            basis_days,
            day_lookup_cache,
            excluded_global_days={49, target_day},
        )
        target_matrix = predict_matrix(
            target_frame,
            basis_days,
            target_global,
            target_days,
            global_weights,
            sector_weights,
            subsector_weights,
        )
        calib_global, calib_days = build_components(
            historical,
            calibration_frame,
            basis_days,
            day_lookup_cache,
            excluded_global_days={49, target_day},
        )
        calib_base = predict_matrix_with_slot_fallback(
            calibration_frame,
            basis_days,
            calib_global,
            calib_days,
            global_weights,
            sector_weights,
            subsector_weights,
        )
        calibrator = fit_calibrator(
            calibration_frame,
            calib_base,
            calibration_frame["demand"].to_numpy(dtype=np.float64),
        )
        fold_predictions = apply_calibrator(target_frame, target_matrix, calibrator)

        geo42 = (
            historical[(historical["day"] == max(target_day - 7, 1)) & (historical["time_min"].isin(NON_OVERLAP_SLOTS))]
            .groupby("geohash")["demand"]
            .mean()
        )
        geo_target = (
            historical[(historical["day"] == target_day) & (historical["time_min"].isin(NON_OVERLAP_SLOTS))]
            .groupby("geohash")["demand"]
            .mean()
        )
        fold_geo_residual = {str(key): float(value) for key, value in geo_target.subtract(geo42, fill_value=0.0).items()}
        fold_predictions = apply_geo_residual(fold_predictions, target_frame, fold_geo_residual)
        fold_score = float(
            r2_score(target_frame["demand"].to_numpy(dtype=np.float64), fold_predictions) * 100.0
        )
        fold_results.append(
            {
                "day": int(target_day),
                "rows": int(len(target_frame)),
                "basis_days": [int(day) for day in basis_days],
                "score": fold_score,
            }
        )

    scores = [float(item["score"]) for item in fold_results]
    return {
        "validation_type": "historical_non_overlap_proxy_folds",
        "held_out_days": VALIDATION_DAYS,
        "folds": fold_results,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def write_validation_summary(summary: dict[str, object]) -> None:
    """Persist the offline validation summary next to the submission bundle."""
    VALIDATION_PATH.write_text(json.dumps(summary, indent=2))


def main() -> None:
    """Train or load the release artifact, then write submission.csv."""
    artifact = load_model_artifact()
    availability = resolve_input_paths()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    if availability.missing_labels:
        emit("Dataset files not found.", log_lines)
        if artifact is not None:
            emit(f"Cached artifact available  : {MODEL_PATH.name}", log_lines)
            emit("Training is not required.", log_lines)
            emit("Inference still needs these data files:", log_lines)
        else:
            emit("No cached artifact is available.", log_lines)
            emit("Training and inference need these data files:", log_lines)
        for label in availability.missing_labels:
            emit(f" - {label}", log_lines)
        emit('Optional dataset source    : kagglehub.dataset_download("kweklydia5/grabtrafficdata")', log_lines)
        RUN_LOG_PATH.write_text("\n".join(log_lines) + "\n")
        return

    historical, competition_train, raw_test = load_inputs()
    if artifact is None:
        artifact = train_model_artifact(historical, competition_train, raw_test)
        save_model_artifact(artifact)
    if not VALIDATION_PATH.exists():
        write_validation_summary(evaluate_proxy_folds(historical, competition_train))

    submission = run_inference(artifact, historical, raw_test)
    submission.to_csv(SUBMISSION_PATH, index=False)

    emit(f"Artifact file            : {MODEL_PATH.name}", log_lines)
    emit(f"Submission file          : {SUBMISSION_PATH.name}", log_lines)
    emit(f"Rows written             : {len(submission):,}", log_lines)
    emit(f"Missing predictions      : {int(submission['demand'].isna().sum())}", log_lines)
    emit(f"Basis days               : {artifact.metrics['basis_days']}", log_lines)
    emit(f"Proxy rows               : {artifact.metrics['proxy_rows']}", log_lines)
    emit(f"Calibration rows         : {artifact.metrics['competition_calibration_rows']}", log_lines)
    emit(f"Residual alpha           : {artifact.metrics['residual_alpha']:.2f}", log_lines)
    emit(f"Boundary 1               : {artifact.metrics['lower_boundary']:.6f}", log_lines)
    emit(f"Boundary 2               : {artifact.metrics['upper_boundary']:.6f}", log_lines)
    if VALIDATION_PATH.exists():
        summary = json.loads(VALIDATION_PATH.read_text())
        emit(f"Proxy fold mean score    : {summary['mean_score']:.6f}%", log_lines)
        emit(f"Proxy fold median score  : {summary['median_score']:.6f}%", log_lines)
    if LOCAL_SCORE_PATH.exists():
        target = pd.read_csv(LOCAL_SCORE_PATH).sort_values("Index")
        scored = float(r2_score(target["demand"].to_numpy(dtype=np.float64), submission["demand"].to_numpy(dtype=np.float64)) * 100.0)
        emit(f"Local score              : {scored:.6f}%", log_lines)
    RUN_LOG_PATH.write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
