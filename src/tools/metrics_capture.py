"""Parse experiment result files into CapturedMetric rows without fabricating values."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

from src.state import CapturedMetric


def _coerce_metric_value(raw: object) -> float | str:
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    try:
        return float(text)
    except ValueError:
        return text


def _as_float_sequence(raw: object) -> list[float] | None:
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        try:
            raw = raw.tolist()
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if not isinstance(raw, (list, tuple)):
        return None
    values: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            item = item[0]
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return values or None


def _stamp(
    metric: CapturedMetric,
    *,
    default_benchmark: str,
    default_algorithm: str,
) -> CapturedMetric:
    benchmark = metric.benchmark.strip() or default_benchmark
    algorithm = metric.algorithm.strip() or default_algorithm
    return metric.model_copy(update={"benchmark": benchmark, "algorithm": algorithm})


def _from_mapping(
    payload: dict,
    source: str,
    default_benchmark: str = "",
    default_algorithm: str = "",
) -> list[CapturedMetric]:
    metrics: list[CapturedMetric] = []
    payload_algo = str(payload.get("algo_name") or payload.get("algorithm") or "").strip()
    algorithm = default_algorithm or payload_algo

    if "aggregates" in payload and isinstance(payload["aggregates"], list):
        for item in payload["aggregates"]:
            if not isinstance(item, dict):
                continue
            metric_name = str(item.get("metric_name", "")).strip()
            if not metric_name:
                continue
            value = item.get("mean", item.get("value"))
            if value is None:
                continue
            metrics.append(
                CapturedMetric(
                    benchmark=str(item.get("benchmark", default_benchmark)).strip(),
                    algorithm=str(item.get("algorithm", algorithm)).strip(),
                    metric_name=metric_name,
                    value=_coerce_metric_value(value),
                    source=str(item.get("source_file_path") or source),
                )
            )
        return metrics

    if "metrics" in payload and isinstance(payload["metrics"], list):
        return _from_list(payload["metrics"], source, default_benchmark, algorithm)

    # BO-style pickle/dict dumps (e.g. BE-CBO) store objective trajectories under Y.
    y_values = _as_float_sequence(payload.get("Y"))
    if y_values:
        benchmark = default_benchmark or str(payload.get("fun_name", "")).strip()
        metrics.append(
            CapturedMetric(
                benchmark=benchmark,
                algorithm=algorithm,
                metric_name="final_objective",
                value=float(y_values[-1]),
                source=source,
            )
        )
        metrics.append(
            CapturedMetric(
                benchmark=benchmark,
                algorithm=algorithm,
                metric_name="best_objective",
                value=float(min(y_values)),
                source=source,
            )
        )
        metrics.append(
            CapturedMetric(
                benchmark=benchmark,
                algorithm=algorithm,
                metric_name="n_evaluations",
                value=float(len(y_values)),
                source=source,
            )
        )
        return metrics

    for metric_name, value in payload.items():
        name = str(metric_name).strip()
        if not name or name in {
            "paper_id",
            "experiment_matrix",
            "paper_reported_results",
            "algo_name",
            "algorithm",
            "fun_name",
        }:
            continue
        if isinstance(value, (dict, list)):
            continue
        metrics.append(
            CapturedMetric(
                benchmark=default_benchmark,
                algorithm=algorithm,
                metric_name=name,
                value=_coerce_metric_value(value),
                source=source,
            )
        )
    return metrics


def _from_list(
    payload: list,
    source: str,
    default_benchmark: str = "",
    default_algorithm: str = "",
) -> list[CapturedMetric]:
    metrics: list[CapturedMetric] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        metric_name = str(item.get("metric_name", "")).strip()
        if not metric_name:
            continue
        if "value" not in item:
            continue
        metrics.append(
            CapturedMetric(
                benchmark=str(item.get("benchmark", default_benchmark)).strip(),
                algorithm=str(item.get("algorithm", default_algorithm)).strip(),
                metric_name=metric_name,
                value=_coerce_metric_value(item.get("value")),
                source=str(item.get("source") or source),
            )
        )
    return metrics


def _from_csv(
    path: Path,
    source: str,
    default_benchmark: str = "",
    default_algorithm: str = "",
) -> list[CapturedMetric]:
    metrics: list[CapturedMetric] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        fields = {name.lower(): name for name in reader.fieldnames}
        metric_key = fields.get("metric_name") or fields.get("metric") or fields.get("name")
        value_key = fields.get("value") or fields.get("mean")
        benchmark_key = fields.get("benchmark")
        algorithm_key = fields.get("algorithm") or fields.get("algo")
        if metric_key is None or value_key is None:
            return []
        for row in reader:
            metric_name = str(row.get(metric_key, "")).strip()
            if not metric_name:
                continue
            raw_value = row.get(value_key)
            if raw_value is None or str(raw_value).strip() == "":
                continue
            benchmark = default_benchmark
            if benchmark_key is not None:
                benchmark = str(row.get(benchmark_key, default_benchmark)).strip()
            algorithm = default_algorithm
            if algorithm_key is not None:
                algorithm = str(row.get(algorithm_key, default_algorithm)).strip()
            metrics.append(
                CapturedMetric(
                    benchmark=benchmark,
                    algorithm=algorithm,
                    metric_name=metric_name,
                    value=_coerce_metric_value(raw_value),
                    source=source,
                )
            )
    return metrics


def _from_pickle(
    path: Path,
    source: str,
    default_benchmark: str = "",
    default_algorithm: str = "",
) -> list[CapturedMetric]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return _from_mapping(payload, source, default_benchmark, default_algorithm)
    return []


def resolve_results_artifact(results_path: Path) -> Path | None:
    """Resolve a plan results_path stem/dir to an existing metrics artifact."""
    if results_path.is_file():
        return results_path
    candidates: list[Path] = [
        results_path.with_suffix(".pkl"),
        results_path.with_suffix(".json"),
        results_path.with_suffix(".csv"),
        Path(f"{results_path}.pkl"),
        Path(f"{results_path}.json"),
        Path(f"{results_path}.csv"),
    ]
    if results_path.is_dir():
        candidates.extend(sorted(results_path.glob("*.json")))
        candidates.extend(sorted(results_path.glob("*.csv")))
        candidates.extend(sorted(results_path.glob("*.pkl")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_metrics_from_path(
    results_path: Path,
    *,
    default_benchmark: str = "",
    default_algorithm: str = "",
) -> tuple[list[CapturedMetric], str | None]:
    """Load metrics from a JSON, CSV, or pickle results file. Never invents values."""
    resolved = resolve_results_artifact(results_path)
    if resolved is None:
        return [], f"Results path does not exist: {results_path}"

    source = str(resolved)
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".csv":
            metrics = _from_csv(resolved, source, default_benchmark, default_algorithm)
        elif suffix == ".pkl":
            metrics = _from_pickle(resolved, source, default_benchmark, default_algorithm)
        else:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metrics = _from_mapping(payload, source, default_benchmark, default_algorithm)
            elif isinstance(payload, list):
                metrics = _from_list(payload, source, default_benchmark, default_algorithm)
            else:
                return [], f"Results file at {resolved} must be JSON object/list, CSV, or pickle dict."
    except (OSError, json.JSONDecodeError, csv.Error, UnicodeDecodeError, pickle.UnpicklingError) as exc:
        return [], f"Failed to parse results at {resolved}: {exc}"

    metrics = [
        _stamp(item, default_benchmark=default_benchmark, default_algorithm=default_algorithm)
        for item in metrics
    ]
    if not metrics:
        return [], f"Results file at {resolved} contained no metric rows."
    return metrics, None


def metric_identity_key(item: CapturedMetric) -> tuple[str, str, str]:
    """Unique key for one matrix cell: benchmark × algorithm × metric."""
    return (
        item.benchmark.strip().lower(),
        item.algorithm.strip().lower(),
        item.metric_name.strip().lower(),
    )


def merge_unique_metrics(
    existing: list[CapturedMetric],
    incoming: list[CapturedMetric],
) -> list[CapturedMetric]:
    """Append incoming metrics, replacing rows with the same (benchmark, algorithm, metric_name)."""
    index = {metric_identity_key(item): item for item in existing}
    for item in incoming:
        index[metric_identity_key(item)] = item
    return list(index.values())
