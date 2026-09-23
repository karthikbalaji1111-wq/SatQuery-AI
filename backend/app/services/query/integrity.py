"""Relational integrity for a client-supplied execution result.

``POST /query/analyze`` accepts a whole :class:`QueryExecutionResult` in the
request body. Pydantic proves it is well-SHAPED: the right fields, the right
types, dates that parse. It cannot prove the parts agree with each other, and
they are exactly the parts an analysis then trusts:

    structural validity  !=  provenance integrity

A body can name a scene that appears in no candidate list, report twelve scenes
while carrying one, attribute a Sentinel-2 window to a modality the intent never
requested, or claim a time window the intent does not contain. Every one of
those parses. The analysis would then read pixels for a scene nobody discovered
and report measurements under a window nobody asked for - and the export, the
evidence panel and any grounded prose would carry that provenance onward as
though the server had established it.

**What this does NOT claim.** It cannot verify that a scene exists, that its
pixels are what the body says, or that discovery ever happened; that would
require re-running discovery, and re-running it would make the endpoint a
different endpoint. What it establishes is narrower and still worth having:
the object is SELF-CONSISTENT, and consistent WITH THE INTENT it claims to
answer. A fabricated result must at least be fabricated coherently, and the
incoherent cases - which is what a mistake looks like - are refused with a
reason rather than analysed.

**Periods bind; labels describe.** The check deliberately draws that line. A
window's PERIOD must be one the intent requested, because attributing a
measurement to dates nobody asked about answers a different question and every
downstream record carries that attribution onward. A window's LABEL is a name
for a requested window, so it must come from this system's own vocabulary and,
WHEN THE INTENT ASSIGNS IT, must mean what the intent says - a "baseline"
carrying the target's dates inverts a comparison. A label the intent does not
assign, over a period it did request, is merely a loose name and is accepted:
refusing it would reject coherent results to enforce a naming convention, which
is not what integrity is for.

Pure: no network, no clock, no service. Raises :class:`ValueError`, so the API
layer renders it in the one validation envelope.
"""

from __future__ import annotations

import re

from app.services.query.schemas import (
    QueryExecutionResult,
    expand_windows,
)

#: Every window label this system produces. A label outside this vocabulary was
#: invented by whoever sent the body.
_CANONICAL_LABEL = re.compile(r"single|baseline|target|series\[\d+\]")


def _fail(reason: str) -> None:
    raise ValueError(f"the execution result is internally inconsistent: {reason}")


def validate_execution_integrity(execution: QueryExecutionResult) -> None:
    """Check a result against itself and against its own intent.

    Returns ``None`` when every relation holds; raises :class:`ValueError`
    naming the FIRST violation otherwise. The message names the relation that
    failed, never the offending value - a body may carry anything, and echoing
    it back is how a request's own contents return to its sender.
    """

    intent = execution.plan.intent
    requested = set(intent.modalities)
    expected = dict(expand_windows(intent))
    # `TimeRange` is a model, so membership is by value - which is what matters
    # here: the same dates under a different object are the same period.
    periods = list(expected.values())

    seen: set[tuple[str, str]] = set()
    for window in execution.windows:
        # --- the window belongs to this request ---------------------------- #
        if window.modality not in requested:
            _fail(
                f"a {window.modality} window is present, but the intent "
                f"requested {sorted(requested)}"
            )

        key = (window.modality, window.label)
        if key in seen:
            _fail(f"the {window.modality} window {window.label!r} appears twice")
        seen.add(key)

        if _CANONICAL_LABEL.fullmatch(window.label) is None:
            _fail(
                f"the window label {window.label!r} is not one this system "
                "produces"
            )

        if window.time_range not in periods:
            # THE binding check. What must never be fabricated is the PERIOD an
            # analysis is attributed to: a measurement reported against dates
            # the user never asked about is a different answer to a different
            # question, and every downstream record would carry it onward.
            _fail(
                f"the {window.modality} window {window.label!r} covers a period "
                "the intent never requested"
            )

        if window.label in expected and window.time_range != expected[window.label]:
            # A label the intent DOES assign must mean what the intent says it
            # means: a window labelled "baseline" carrying the target's dates
            # inverts a comparison while every field still parses.
            _fail(
                f"the window {window.label!r} carries a period the intent does "
                "not assign to that label"
            )

        # --- the window agrees with itself --------------------------------- #
        if window.scene_count < len(window.scenes):
            # Only the incoherent direction is refused. A window may carry FEWER
            # scenes than it reports - a catalog page is a bounded subset of the
            # matches, which is a real property of STAC discovery rather than a
            # defect - but it cannot carry more scenes than it found.
            _fail(
                f"the {window.modality} window {window.label!r} reports "
                f"{window.scene_count} scene(s) while carrying "
                f"{len(window.scenes)}"
            )

        # The selected scene is what every later read is keyed on, and nothing
        # else checks that it was ever discovered.
        if window.selected_scene_id is not None and all(
            scene.id != window.selected_scene_id for scene in window.scenes
        ):
            _fail(
                f"the {window.modality} window {window.label!r} selected a "
                "scene that is not among the scenes it returned"
            )

        if window.imagery is not None:
            if window.selected_scene_id is None:
                _fail(
                    f"the {window.modality} window {window.label!r} carries "
                    "imagery without having selected a scene"
                )
            elif window.imagery.scene_id != window.selected_scene_id:
                _fail(
                    f"the {window.modality} window {window.label!r} carries "
                    "imagery for a different scene than the one it selected"
                )

        # --- a failed window is empty -------------------------------------- #
        if window.error is not None and (
            window.scenes or window.selected_scene_id is not None
        ):
            _fail(
                f"the {window.modality} window {window.label!r} reports a "
                "discovery failure while also carrying results"
            )

    # --- the modalities agree with the windows ----------------------------- #
    executed = set(execution.executed_modalities)
    if not executed <= requested:
        _fail(
            "executed_modalities names a modality the intent did not request"
        )
