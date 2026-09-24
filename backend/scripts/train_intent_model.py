"""Train, evaluate and export the SatQuery intent model.

    uv run python scripts/train_intent_model.py [--baseline OLD.json.gz]   (from backend/)

Reproducible end to end: the same dataset file and this script produce the
same artifact bytes. Nothing is downloaded; scikit-learn is a DEV dependency
used here and in the equivalence test only - production reads the exported
JSON with numpy (``app/services/agent/intent_model.py``).

Protocol
--------
1. Load ``data/intent/satquery_intents_v2.jsonl`` and validate every row.
2. Mask places and dates with the deterministic reader (``classifier_text``),
   exactly as inference does, so the model learns operations, not cities.
3. Group near-duplicates and cross-label twins into families (see
   ``cluster``) and split BY FAMILY, keeping each label near 70/15/15, into
   train / validation / test with a stable hash. No family straddles a split.
4. Choose C by grouped 5-fold cross-validation on train (macro F1).
5. Fit the final model on train only.
6. Choose the confidence threshold on out-of-fold train predictions plus the
   validation split: the lowest threshold at which at least 99% of accepted
   predictions are correct, and never below ``THRESHOLD_FLOOR`` (0.90). The
   test split takes no part in any choice.
7. Evaluate once on test: model metrics, then the full decision rule
   (model + rules + guards) against the rules alone.
8. Export the pipeline to JSON, reload it through the production loader, and
   refuse to write anything unless it reproduces scikit-learn's probabilities.
9. Evaluate the challenge sets (``CHALLENGES``). ``satquery_intents_v2_
   challenge.jsonl`` was written before any v2 training and influences
   nothing; the v1 challenge set's misses informed dataset v2, so it is
   reported as development data. With ``--baseline <artifact>`` the previous
   model is evaluated on the same sets through the same rules (before -> after).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.agent.intent_model import (  # noqa: E402
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_ARTIFACT,
    INTENT_LABELS,
    IntentModelError,
    parse_artifact,
    read_artifact,
)
from app.services.agent.intent_router import route, route_operation  # noqa: E402
from app.services.agent.interpretation import (  # noqa: E402
    ClarificationRequiredError,
    classifier_text,
)

DATASET = BACKEND / "data" / "intent" / "satquery_intents_v2.jsonl"
#: (name, path, role). Only the FIRST is held out from every choice. The v1
#: challenge set was evaluated once against v1; its misses were then read and
#: informed dataset v2, so for v2 it is development data and is labelled so.
CHALLENGES = (
    ("challenge_v2", BACKEND / "data" / "intent" / "satquery_intents_v2_challenge.jsonl",
     "held out: written before any v2 training, evaluated once"),
    ("challenge_v1", BACKEND / "data" / "intent" / "satquery_intents_v1_challenge.jsonl",
     "development: its v1 misses were inspected and informed dataset v2"),
)
REPORT_JSON = BACKEND / "data" / "intent" / "evaluation_v2.json"
REPORT_MD = BACKEND / "data" / "intent" / "evaluation_v2.md"
MODEL_VERSION = "satquery-intent-v2"

SEED = "satquery-intent-v2"
SPLIT = (0.70, 0.15, 0.15)
NEAR_DUPLICATE_JACCARD = 0.75
TWIN_JACCARD = 0.55
C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 .. 0.95
TARGET_ACCEPTED_PRECISION = 0.99
#: The shipped threshold is never below this, whatever the calibration finds.
#: A lower threshold would raise coverage by acting on less certain predictions;
#: coverage is to come from better data, not from a looser rule.
THRESHOLD_FLOOR = 0.90
#: "Today" for the full-pipeline evaluation, pinned so the report is
#: reproducible: a future-date refusal must not depend on when it is re-run.
EVALUATION_DATE = date(2026, 9, 24)
WORD = {"ngram_range": (1, 2), "min_df": 1}
CHAR = {"ngram_range": (2, 5), "min_df": 2}
SIGNIFICANT_DIGITS = 9


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_dataset() -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "dataset ids must be unique"
    for row in rows:
        assert set(row) == {"id", "text", "intent"}, row
        assert row["intent"] in INTENT_LABELS, row
        assert row["text"].strip(), row
    return rows


def _normalised(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def _shingles(text: str, n: int = 4) -> set[str]:
    padded = f" {text} "
    return {padded[i : i + n] for i in range(max(1, len(padded) - n + 1))}


def cluster(rows: list[dict]) -> list[int]:
    """Group near-duplicates AND cross-label twins; return a family id per row.

    Two rows share a family when their masked texts are near-identical (same
    label, Jaccard >= NEAR_DUPLICATE_JACCARD) or when their RAW texts are close
    whatever the label (Jaccard >= TWIN_JACCARD). The second rule matters: the
    dataset deliberately contains contrastive twins - one skeleton written once
    per analysis ("<verb> vegetation/water/built-up around <place> in <date>").
    Splitting twins apart would put a test sentence's own skeleton in training
    under OTHER labels, which measures skeleton memory instead of whether the
    operation word is read. A family therefore never straddles a split.
    """

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    masked = [_shingles(_normalised(row["masked"])) for row in rows]
    raw = [_shingles(_normalised(row["text"])) for row in rows]

    def jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            same_label = rows[a]["intent"] == rows[b]["intent"]
            if (same_label and jaccard(masked[a], masked[b]) >= NEAR_DUPLICATE_JACCARD) or (
                jaccard(raw[a], raw[b]) >= TWIN_JACCARD
            ):
                parent[find(a)] = find(b)
    return [find(i) for i in range(len(rows))]


def _stable(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def split(rows: list[dict], clusters: list[int]) -> list[str]:
    """Assign whole families to train/validation/test, keeping each label's
    proportions as close to 70/15/15 as whole families allow."""

    names = ("train", "validation", "test")
    totals = Counter(row["intent"] for row in rows)
    target = {
        name: {label: totals[label] * fraction for label in totals}
        for name, fraction in zip(names, SPLIT, strict=True)
    }
    filled = {name: Counter() for name in names}
    families: dict[int, list[int]] = defaultdict(list)
    for i, family in enumerate(clusters):
        families[family].append(i)
    ordered = sorted(families.values(), key=lambda members: _stable(rows[members[0]]["id"]))
    # Largest families first, so the small ones can even out the proportions.
    ordered.sort(key=len, reverse=True)
    assignment = [""] * len(rows)
    for members in ordered:
        labels = Counter(rows[i]["intent"] for i in members)
        best = min(
            names,
            key=lambda name: (
                max((filled[name][lab] + n) / target[name][lab] for lab, n in labels.items()),
                names.index(name),
            ),
        )
        for i in members:
            assignment[i] = best
        filled[best].update(labels)
    return assignment


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", lowercase=True, sublinear_tf=True,
                ngram_range=WORD["ngram_range"], min_df=WORD["min_df"],
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", lowercase=True, sublinear_tf=True,
                ngram_range=CHAR["ngram_range"], min_df=CHAR["min_df"],
            )),
        ])),
        ("classifier", LogisticRegression(C=c, max_iter=10_000)),
    ])


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def export(pipeline: Pipeline, threshold: float, metadata: dict) -> dict:
    union = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    blocks = {}
    for name, vectorizer in union.transformer_list:
        vocabulary = [
            term for term, _ in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])
        ]
        blocks[name] = {
            "analyzer": vectorizer.analyzer,
            "ngram_range": list(vectorizer.ngram_range),
            "vocabulary": vocabulary,
            "idf": [_round(v) for v in vectorizer.idf_],
        }
    return {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "labels": [str(label) for label in classifier.classes_],
        "threshold": threshold,
        "features": blocks,
        "classifier": {
            "coef": [[_round(v) for v in row] for row in classifier.coef_],
            "intercept": [_round(v) for v in classifier.intercept_],
        },
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(INTENT_LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "rows_true_columns_predicted": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
        },
    }


def threshold_table(correct: list[bool], confidence: list[float]) -> list[dict]:
    table = []
    for t in THRESHOLD_GRID:
        accepted = [ok for ok, c in zip(correct, confidence, strict=True) if c >= t]
        table.append({
            "threshold": t,
            "accepted": len(accepted),
            "coverage": len(accepted) / len(correct),
            "accepted_errors": accepted.count(False),
            "accepted_precision": (accepted.count(True) / len(accepted)) if accepted else 1.0,
        })
    return table


def choose_threshold(table: list[dict]) -> float:
    for row in table:
        if row["accepted_precision"] >= TARGET_ACCEPTED_PRECISION:
            return row["threshold"]
    return THRESHOLD_GRID[-1]


_EXPECTED = {
    "NDVI": (["ndvi"], False),
    "NDWI": (["ndwi"], False),
    "NDBI": (["ndbi"], False),
    "SAR_BACKSCATTER": (["sar_backscatter"], False),
    "TRUE_COLOR": (["imagery"], False),
    "TEMPORAL_NDWI": (["temporal_ndwi"], True),
}


def system_outcome(text: str, label: str, classifier) -> tuple[str, str]:
    """correct / clarified / WRONG for the full decision rule on one question."""

    try:
        chosen, comparison, decision = route_operation(text, classifier)
    except ClarificationRequiredError as exc:
        if label in ("CLARIFICATION", "UNSUPPORTED"):
            return "correct", f"clarify:{exc.clarification.reason}"
        return "clarified", f"clarify:{exc.clarification.reason}"
    detail = f"execute:{'+'.join(chosen)}{' compare' if comparison else ''}"
    if label in ("CLARIFICATION", "UNSUPPORTED"):
        return "WRONG", detail
    return ("correct" if (chosen, comparison) == _EXPECTED[label] else "WRONG"), detail


def system_outcome_full(text: str, label: str, classifier) -> tuple[str, str]:
    """The same judgement over the FULL pipeline: operation, guards, place, dates.

    What a user actually gets. A question whose operation was misread can still
    be stopped later - e.g. two unlinked dates in a single-period question are
    asked back - and only an EXECUTED wrong operation is a wrong answer.
    """

    try:
        interpretation, _ = route(text, classifier, today=EVALUATION_DATE)
    except ClarificationRequiredError as exc:
        if label in ("CLARIFICATION", "UNSUPPORTED"):
            return "correct", f"clarify:{exc.clarification.reason}"
        return "clarified", f"clarify:{exc.clarification.reason}"
    chosen, comparison = list(interpretation.analyses), interpretation.comparison
    detail = f"execute:{'+'.join(chosen)}{' compare' if comparison else ''}"
    if label in ("CLARIFICATION", "UNSUPPORTED"):
        return "WRONG", detail
    return ("correct" if (chosen, comparison) == _EXPECTED[label] else "WRONG"), detail


def system_summary(rows: list[dict], classifier, judge=system_outcome) -> dict:
    outcomes = [judge(row["text"], row["intent"], classifier) for row in rows]
    counts = Counter(outcome for outcome, _ in outcomes)
    return {
        "correct": counts["correct"],
        "clarified_safely": counts["clarified"],
        "wrong_operation_executed": counts["WRONG"],
        "examples": [
            {"id": row["id"], "text": row["text"], "intent": row["intent"],
             "outcome": outcome, "detail": detail}
            for row, (outcome, detail) in zip(rows, outcomes, strict=True)
            if outcome != "correct"
        ],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def evaluate_challenge(
    path: Path, role: str, classifier, train_shingles: list[set[str]] | None
) -> dict:
    """One challenge set through the model alone and through the full rule.

    ``train_shingles`` gives each item's closest training match; it is ``None``
    for a baseline artifact, whose training rows are not this run's.
    """

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["masked"] = classifier_text(row["text"])
    predictions = [classifier.predict(r["masked"]) for r in rows]
    threshold = classifier.threshold
    correct = [p.label == r["intent"] for p, r in zip(predictions, rows, strict=True)]
    report = {
        "path": str(path.relative_to(BACKEND)),
        "role": role,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": metrics([r["intent"] for r in rows], [p.label for p in predictions]),
        "at_threshold": threshold_table(correct, [p.confidence for p in predictions])[
            THRESHOLD_GRID.index(threshold)
        ],
        "system": {
            "model_plus_rules": system_summary(rows, classifier),
            "rules_only": system_summary(rows, None),
            "full_pipeline_model_plus_rules": system_summary(
                rows, classifier, system_outcome_full
            ),
            "full_pipeline_rules_only": system_summary(rows, None, system_outcome_full),
        },
        "misclassified": [
            {"id": r["id"], "text": r["text"], "intent": r["intent"], "predicted": p.label,
             "confidence": round(p.confidence, 4), "acted_on": p.confidence >= threshold}
            for r, p in zip(rows, predictions, strict=True)
            if p.label != r["intent"]
        ],
    }
    if train_shingles is not None:

        def closest(text: str) -> float:
            mine = _shingles(_normalised(text))
            return max(len(mine & other) / len(mine | other) for other in train_shingles)

        nearest = [closest(r["masked"]) for r in rows]
        report["max_masked_jaccard_to_train"] = max(nearest)
        report["items_at_or_above_near_duplicate_threshold"] = sum(
            n >= NEAR_DUPLICATE_JACCARD for n in nearest
        )
    return report


def main(baseline: Path | None = None) -> None:
    baseline_model = read_artifact(baseline) if baseline is not None else None
    raw = DATASET.read_bytes()
    rows = load_dataset()
    for row in rows:
        row["masked"] = classifier_text(row["text"])
    clusters = cluster(rows)
    splits = split(rows, clusters)
    for row, name, group in zip(rows, splits, clusters, strict=True):
        row["split"], row["cluster"] = name, group

    part = {
        name: [r for r in rows if r["split"] == name]
        for name in ("train", "validation", "test")
    }
    for name in ("validation", "test"):
        assert not {r["cluster"] for r in part[name]} & {r["cluster"] for r in part["train"]}
    texts = {name: [r["masked"] for r in part[name]] for name in part}
    y = {name: [r["intent"] for r in part[name]] for name in part}

    # --- 4. C by grouped CV on train ---------------------------------------
    folds = GroupKFold(n_splits=5)
    groups = [r["cluster"] for r in part["train"]]
    cv_scores: dict[float, float] = {}
    oof: dict[float, tuple[list[str], list[float]]] = {}
    for c in C_GRID:
        predicted = [""] * len(texts["train"])
        confident = [0.0] * len(texts["train"])
        for train_idx, hold_idx in folds.split(texts["train"], y["train"], groups):
            model = make_pipeline(c).fit(
                [texts["train"][i] for i in train_idx], [y["train"][i] for i in train_idx]
            )
            probabilities = model.predict_proba([texts["train"][i] for i in hold_idx])
            for i, p in zip(hold_idx, probabilities, strict=True):
                predicted[i] = str(model.classes_[int(np.argmax(p))])
                confident[i] = float(np.max(p))
        cv_scores[c] = f1_score(y["train"], predicted, average="macro")
        oof[c] = (predicted, confident)
    best_c = max(C_GRID, key=lambda c: (round(cv_scores[c], 6), -c))

    # --- 5. final model -------------------------------------------------------
    pipeline = make_pipeline(best_c).fit(texts["train"], y["train"])

    # --- 6. threshold on out-of-fold + validation -----------------------------
    val_proba = pipeline.predict_proba(texts["validation"])
    val_pred = [str(pipeline.classes_[int(np.argmax(p))]) for p in val_proba]
    val_conf = [float(np.max(p)) for p in val_proba]
    oof_pred, oof_conf = oof[best_c]
    calibration_correct = [
        *(p == t for p, t in zip(oof_pred, y["train"], strict=True)),
        *(p == t for p, t in zip(val_pred, y["validation"], strict=True)),
    ]
    calibration_conf = [*oof_conf, *val_conf]
    calibration = threshold_table(calibration_correct, calibration_conf)
    threshold = max(THRESHOLD_FLOOR, choose_threshold(calibration))

    # --- 7. test, once ----------------------------------------------------------
    test_proba = pipeline.predict_proba(texts["test"])
    test_pred = [str(pipeline.classes_[int(np.argmax(p))]) for p in test_proba]
    test_conf = [float(np.max(p)) for p in test_proba]
    test_correct = [p == t for p, t in zip(test_pred, y["test"], strict=True)]

    # --- 8. export and verify ---------------------------------------------------
    metadata = {
        "dataset": {
            "path": str(DATASET.relative_to(BACKEND)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "examples": len(rows),
            "per_class": dict(sorted(Counter(r["intent"] for r in rows).items())),
            "split_counts": {name: len(part[name]) for name in part},
            "families": len(set(clusters)),
            "largest_family": max(Counter(clusters).values()),
            "family_rule": (
                f"same label and masked Jaccard >= {NEAR_DUPLICATE_JACCARD}, or any "
                f"label and raw Jaccard >= {TWIN_JACCARD} (char 4-grams)"
            ),
        },
        "training": {
            "script": "scripts/train_intent_model.py",
            "features": {
                "word": {"analyzer": "word", **WORD, "sublinear_tf": True},
                "char": {"analyzer": "char_wb", **CHAR, "sublinear_tf": True},
            },
            "classifier": "LogisticRegression (multinomial, lbfgs)",
            "c_grid": list(C_GRID),
            "c_selected": best_c,
            "c_cv_macro_f1": {str(c): cv_scores[c] for c in C_GRID},
            "threshold_rule": (
                f"lowest of {THRESHOLD_GRID[0]}..{THRESHOLD_GRID[-1]} with accepted "
                f"precision >= {TARGET_ACCEPTED_PRECISION} on out-of-fold train "
                f"plus validation predictions, never below {THRESHOLD_FLOOR}"
            ),
            "threshold_calibrated": choose_threshold(calibration),
            "scikit_learn": sklearn.__version__,
            "input": "classifier_text(): places -> PLACE, dates -> DATE",
        },
    }
    document = export(pipeline, threshold, metadata)
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    classifier = parse_artifact(json.loads(payload))

    every = [r["masked"] for r in rows]
    reference = pipeline.predict_proba(every)
    exported = np.vstack([classifier.probabilities(text) for text in every])
    order = [list(pipeline.classes_).index(label) for label in classifier.labels]
    max_diff = float(np.max(np.abs(reference[:, order] - exported)))
    same_argmax = bool(
        np.all(np.argmax(reference[:, order], axis=1) == np.argmax(exported, axis=1))
    )
    if max_diff > 1e-6 or not same_argmax:
        raise IntentModelError(
            f"exported model diverges from scikit-learn (max diff {max_diff:.2e})"
        )

    artifact_bytes = gzip.compress(payload, compresslevel=9, mtime=0)
    DEFAULT_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ARTIFACT.write_bytes(artifact_bytes)

    # latency and memory, through the production loader
    tracemalloc.start()
    loaded = read_artifact(DEFAULT_ARTIFACT)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings = []
    for text in texts["test"] * 5:
        start = time.perf_counter()
        loaded.predict(text)
        timings.append((time.perf_counter() - start) * 1000)
    route_timings = []
    for row in part["test"]:
        start = time.perf_counter()
        with contextlib.suppress(ClarificationRequiredError):
            route_operation(row["text"], loaded)
        route_timings.append((time.perf_counter() - start) * 1000)

    misclassified = [
        {"id": r["id"], "text": r["text"], "masked": r["masked"], "intent": r["intent"],
         "predicted": p, "confidence": round(c, 4),
         "acted_on": c >= threshold}
        for r, p, c, ok in zip(part["test"], test_pred, test_conf, test_correct, strict=True)
        if not ok
    ]
    # --- 9. the challenge sets, evaluated once -----------------------------------
    train_shingles = [_shingles(_normalised(r["masked"])) for r in part["train"]]
    challenges = {
        name: evaluate_challenge(path, role, loaded, train_shingles)
        for name, path, role in CHALLENGES
    }
    baseline = (
        {
            "path": baseline.name,
            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "model_version": baseline_model.version,
            "threshold": baseline_model.threshold,
            "note": (
                "the previous artifact through TODAY's rules and guards, so the "
                "comparison isolates the model; its own training rows may overlap "
                "challenge_v1 only as that set's role states"
            ),
            "challenges": {
                name: evaluate_challenge(path, role, baseline_model, None)
                for name, path, role in CHALLENGES
            },
        }
        if baseline is not None and baseline_model is not None
        else None
    )

    report = {
        "model_version": MODEL_VERSION,
        "artifact": {
            "path": str(DEFAULT_ARTIFACT.relative_to(BACKEND)),
            "bytes": len(artifact_bytes),
            "json_bytes": len(payload),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "features_word": len(document["features"]["word"]["vocabulary"]),
            "features_char": len(document["features"]["char"]["vocabulary"]),
            "peak_load_memory_bytes": peak,
            "sklearn_equivalence_max_abs_diff": max_diff,
            "sklearn_equivalence_same_argmax": same_argmax,
        },
        "latency_ms": {
            "predict_mean": float(np.mean(timings)),
            "predict_p95": float(np.percentile(timings, 95)),
            "route_operation_mean": float(np.mean(route_timings)),
            "route_operation_p95": float(np.percentile(route_timings, 95)),
        },
        "threshold": threshold,
        "threshold_calibration": calibration,
        "validation": metrics(y["validation"], val_pred),
        "test": metrics(y["test"], test_pred),
        "test_at_threshold": threshold_table(test_correct, test_conf)[
            THRESHOLD_GRID.index(threshold)
        ],
        "test_misclassified": misclassified,
        "system_on_test": {
            "model_plus_rules": system_summary(part["test"], loaded),
            "rules_only": system_summary(part["test"], None),
            "full_pipeline_model_plus_rules": system_summary(
                part["test"], loaded, system_outcome_full
            ),
            "full_pipeline_rules_only": system_summary(part["test"], None, system_outcome_full),
        },
        "challenges": challenges,
        "baseline": baseline,
        "metadata": metadata,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(render_markdown(report))
    print(render_markdown(report))


def render_challenge(
    name: str, ch: dict, threshold: float, labels: list[str], short: dict[str, str]
) -> list[str]:
    cm = ch["model"]
    lines = [
        "",
        f"## Challenge set `{name}` - {ch['role']}",
        "",
        f"- `{ch['path']}`, n = {cm['count']}"
        + (
            f"; closest training match: max masked Jaccard "
            f"{ch['max_masked_jaccard_to_train']:.3f}, "
            f"{ch['items_at_or_above_near_duplicate_threshold']} item(s) at or above "
            f"{NEAR_DUPLICATE_JACCARD}"
            if "max_masked_jaccard_to_train" in ch
            else ""
        ),
        f"- Model: accuracy {cm['accuracy']:.4f}; macro F1 {cm['macro_f1']:.4f}; "
        f"weighted F1 {cm['weighted_f1']:.4f}",
        f"- At threshold {threshold}: {ch['at_threshold']['accepted']} acted on "
        f"(coverage {ch['at_threshold']['coverage']:.3f}), "
        f"{ch['at_threshold']['accepted_errors']} wrong",
        "",
        "| class | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in cm["per_class"].items():
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['support']} |"
        )
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
              "| | " + " | ".join(short.get(lab, lab) for lab in labels) + " |",
              "| --- |" + " --- |" * len(labels)]
    for label, row in zip(labels, cm["confusion_matrix"]["rows_true_columns_predicted"],
                          strict=True):
        lines.append(f"| {short.get(label, label)} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "| | correct | clarified safely | WRONG operation executed |",
              "| --- | --- | --- | --- |"]
    for title, key in (
        ("operation only: model + rules", "model_plus_rules"),
        ("operation only: rules only (M5.5)", "rules_only"),
        ("full pipeline: model + rules", "full_pipeline_model_plus_rules"),
        ("full pipeline: rules only (M5.5)", "full_pipeline_rules_only"),
    ):
        sy = ch["system"][key]
        lines.append(
            f"| {title} | {sy['correct']} | {sy['clarified_safely']} | "
            f"{sy['wrong_operation_executed']} |"
        )
    lines += ["", "Misclassified by the model:", ""]
    if not ch["misclassified"]:
        lines.append("None.")
    for m in ch["misclassified"]:
        lines.append(
            f"- `{m['id']}` {m['intent']} -> {m['predicted']} ({m['confidence']}, "
            f"{'ACTED ON' if m['acted_on'] else 'below threshold'}): {m['text']}"
        )
    lines += ["", "Non-correct outcomes, full pipeline, model + rules:", ""]
    for e in ch["system"]["full_pipeline_model_plus_rules"]["examples"]:
        lines.append(f"- `{e['id']}` {e['intent']}: {e['outcome']} ({e['detail']}) - {e['text']}")
    return lines


def render_markdown(report: dict) -> str:
    test = report["test"]
    lines = [
        f"# SatQuery intent model - {report['model_version']}",
        "",
        "Generated by `scripts/train_intent_model.py`. Do not edit by hand.",
        "",
        "## Data",
        "",
        f"- Dataset: `{report['metadata']['dataset']['path']}` "
        f"(sha256 `{report['metadata']['dataset']['sha256'][:16]}...`), "
        f"{report['metadata']['dataset']['examples']} examples",
        f"- Per class: {report['metadata']['dataset']['per_class']}",
        f"- Split (by family): {report['metadata']['dataset']['split_counts']}",
        f"- Families (no family straddles a split): {report['metadata']['dataset']['families']}"
        f" (largest {report['metadata']['dataset']['largest_family']}; rule: "
        f"{report['metadata']['dataset']['family_rule']})",
        "",
        "## Model",
        "",
        "- TF-IDF word 1-2-grams + char_wb 2-5-grams (min_df 2), sublinear TF, "
        "L2 per block -> multinomial LogisticRegression",
        f"- C = {report['metadata']['training']['c_selected']} "
        f"(grouped 5-fold CV macro F1: {report['metadata']['training']['c_cv_macro_f1']})",
        f"- Artifact: `{report['artifact']['path']}`, {report['artifact']['bytes']:,} bytes "
        f"gzip ({report['artifact']['json_bytes']:,} JSON), "
        f"{report['artifact']['features_word']:,} word + "
        f"{report['artifact']['features_char']:,} char features",
        "- numpy vs scikit-learn: max |dp| = "
        f"{report['artifact']['sklearn_equivalence_max_abs_diff']:.2e}, "
        f"same argmax = {report['artifact']['sklearn_equivalence_same_argmax']}",
        f"- Latency: predict mean {report['latency_ms']['predict_mean']:.3f} ms "
        f"(p95 {report['latency_ms']['predict_p95']:.3f}); full operation routing mean "
        f"{report['latency_ms']['route_operation_mean']:.3f} ms",
        f"- Peak memory while loading: {report['artifact']['peak_load_memory_bytes']:,} bytes",
        "",
        "## Held-out test",
        "",
        f"- n = {test['count']}; accuracy {test['accuracy']:.4f}; macro F1 "
        f"{test['macro_f1']:.4f}; weighted F1 {test['weighted_f1']:.4f}",
        "",
        "| class | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in test["per_class"].items():
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['support']} |"
        )
    labels = test["confusion_matrix"]["labels"]
    short = {"SAR_BACKSCATTER": "SAR", "TEMPORAL_NDWI": "T-NDWI", "TRUE_COLOR": "RGB",
             "CLARIFICATION": "CLAR", "UNSUPPORTED": "UNSUP"}
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
              "| | " + " | ".join(short.get(lab, lab) for lab in labels) + " |",
              "| --- |" + " --- |" * len(labels)]
    matrix = test["confusion_matrix"]["rows_true_columns_predicted"]
    for label, row in zip(labels, matrix, strict=True):
        lines.append(f"| {short.get(label, label)} | " + " | ".join(str(v) for v in row) + " |")
    at = report["test_at_threshold"]
    lines += [
        "",
        f"## Threshold: {report['threshold']}",
        "",
        f"Rule: {report['metadata']['training']['threshold_rule']}. The calibration "
        f"alone chose {report['metadata']['training']['threshold_calibrated']}.",
        "",
        "| threshold | accepted | coverage | accepted errors | accepted precision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["threshold_calibration"]:
        lines.append(
            f"| {row['threshold']} | {row['accepted']} | {row['coverage']:.3f} | "
            f"{row['accepted_errors']} | {row['accepted_precision']:.4f} |"
        )
    lines += [
        "",
        f"On test at {report['threshold']}: {at['accepted']} of {test['count']} acted on "
        f"(coverage {at['coverage']:.3f}), {at['accepted_errors']} of them wrong "
        f"(accepted precision {at['accepted_precision']:.4f}).",
        "",
        "## Misclassified test examples",
        "",
    ]
    if not report["test_misclassified"]:
        lines.append("None.")
    for m in report["test_misclassified"]:
        lines.append(
            f"- `{m['id']}` {m['intent']} -> {m['predicted']} ({m['confidence']}, "
            f"{'ACTED ON' if m['acted_on'] else 'below threshold'}): {m['text']}"
        )
    lines += ["", "## Full decision rule on test (operation only)", "",
              "| | correct | clarified safely | WRONG operation executed |",
              "| --- | --- | --- | --- |"]
    for name, key in (
        ("operation only: model + rules", "model_plus_rules"),
        ("operation only: rules only (M5.5)", "rules_only"),
        ("full pipeline: model + rules", "full_pipeline_model_plus_rules"),
        ("full pipeline: rules only (M5.5)", "full_pipeline_rules_only"),
    ):
        s = report["system_on_test"][key]
        lines.append(
            f"| {name} | {s['correct']} | {s['clarified_safely']} | "
            f"{s['wrong_operation_executed']} |"
        )
    lines += ["", "Non-correct outcomes, model + rules:", ""]
    for e in report["system_on_test"]["model_plus_rules"]["examples"]:
        lines.append(f"- `{e['id']}` {e['intent']}: {e['outcome']} ({e['detail']}) - {e['text']}")

    for name, ch in report["challenges"].items():
        lines += render_challenge(name, ch, report["threshold"], labels, short)
    if report["baseline"] is not None:
        b = report["baseline"]
        lines += [
            "",
            f"## Before -> after: {b['model_version']} (threshold {b['threshold']}) -> "
            f"{report['model_version']} (threshold {report['threshold']})",
            "",
            f"Baseline: `{b['path']}` (sha256 `{b['sha256'][:16]}...`) - {b['note']}.",
            "",
            "| set | model | macro F1 | acted on (coverage) | acted on wrong | "
            "full pipeline correct | clarified | WRONG executed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, after in report["challenges"].items():
            for version, ch in ((b["model_version"], b["challenges"][name]),
                                (report["model_version"], after)):
                at = ch["at_threshold"]
                sy = ch["system"]["full_pipeline_model_plus_rules"]
                lines.append(
                    f"| {name} | {version} | {ch['model']['macro_f1']:.4f} | "
                    f"{at['accepted']} ({at['coverage']:.3f}) | {at['accepted_errors']} | "
                    f"{sy['correct']} | {sy['clarified_safely']} | "
                    f"{sy['wrong_operation_executed']} |"
                )
            ro = after["system"]["full_pipeline_rules_only"]
            lines.append(
                f"| {name} | rules only | - | - | - | {ro['correct']} | "
                f"{ro['clarified_safely']} | {ro['wrong_operation_executed']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="a previous artifact to evaluate on the same challenge sets (before -> after)",
    )
    main(parser.parse_args().baseline)
