"""What dates a query may name, and what the product does when it names none.

Two separate concerns, both of which have to be PROVIDER-INDEPENDENT - the same
question must not mean different dates depending on which model planned it.

**Observability bounds** are enforced on `SatQueryIntent` itself, so every path
that builds one inherits them: the Gemini planner, the NVIDIA planner, the
manual form and a direct API call alike. A window that no satellite could have
observed is refused with a reason, rather than being sent to the catalog and
coming back empty - "no scenes found" is indistinguishable from a genuine gap in
coverage and points the reader at the wrong problem.

**The default window** applies when the user names no date. It is a product
default, stated once in `default_time_window()` so both providers are given the
same rule, and rendered into the planning instruction from there.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from app.services.agent.prompts import (
    DEFAULT_LOOKBACK_DAYS,
    _system_instruction,
    default_time_window,
)
from app.services.query.schemas import (
    EARLIEST_OBSERVATION_DATE,
    SatQueryIntent,
    TemporalComparison,
    TimeRange,
)
from pydantic import ValidationError

TODAY = datetime.now(UTC).date()


def _intent(start: date, end: date, **kw: object) -> SatQueryIntent:
    payload: dict[str, object] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [TimeRange(start_date=start, end_date=end)],
        "modalities": ["sentinel-2-optical"],
        "task": "visualize",
    }
    payload.update(kw)
    return SatQueryIntent(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Impossible windows are refused, with the reason.
# --------------------------------------------------------------------------- #


def test_a_window_entirely_in_the_future_is_refused() -> None:
    with pytest.raises(ValidationError) as caught:
        _intent(date(2099, 1, 1), date(2099, 1, 31))

    assert "future" in str(caught.value)


def test_a_window_before_any_satellite_was_acquiring_is_refused() -> None:
    with pytest.raises(ValidationError) as caught:
        _intent(date(1990, 1, 1), date(1990, 1, 31))

    assert EARLIEST_OBSERVATION_DATE.isoformat() in str(caught.value)


def test_the_bound_applies_to_both_halves_of_a_comparison() -> None:
    """A `compare` intent holds its windows in an object, not a list.

    Reading only the list branch would leave the temporal path - the one that
    takes TWO windows - entirely unguarded.
    """

    with pytest.raises(ValidationError):
        SatQueryIntent(
            location_query="Chennai",
            temporal_mode="compare",
            time_windows=TemporalComparison(
                baseline=TimeRange(
                    start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)
                ),
                target=TimeRange(
                    start_date=date(2099, 1, 1), end_date=date(2099, 1, 31)
                ),
            ),
            modalities=["sentinel-2-optical"],
            task="visualize",
        )


# --------------------------------------------------------------------------- #
# Legitimate windows are NOT refused. Without these the bound could be a
# blanket rejection and still pass everything above.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("an ordinary past window", date(2025, 1, 1), date(2025, 1, 31)),
        ("a single day", date(2025, 3, 24), date(2025, 3, 24)),
        ("today", TODAY, TODAY),
        ("the default lookback", TODAY - timedelta(days=30), TODAY),
        # A caller a few hours ahead of UTC must not have "today" refused.
        ("ending tomorrow, clock slack", TODAY - timedelta(days=5),
         TODAY + timedelta(days=1)),
        # Spans the mission epoch: the early part is empty, not impossible.
        ("spanning the first observation", date(2010, 1, 1), date(2020, 1, 1)),
    ],
)
def test_a_legitimate_window_is_accepted(
    label: str, start: date, end: date
) -> None:
    assert _intent(start, end) is not None, label


def test_an_inverted_window_is_still_refused() -> None:
    """Pre-existing rule; pinned here so the new bounds cannot displace it."""

    with pytest.raises(ValidationError):
        _intent(date(2025, 12, 31), date(2025, 1, 1))


# --------------------------------------------------------------------------- #
# The no-date default: one rule, both providers.
# --------------------------------------------------------------------------- #


def test_the_default_window_is_a_bounded_recent_lookback() -> None:
    today, start = default_time_window()

    assert today == datetime.now(UTC).date()
    assert (today - start).days == DEFAULT_LOOKBACK_DAYS
    assert start < today


def test_the_default_window_is_itself_a_valid_intent() -> None:
    """The product default must satisfy the product's own bounds."""

    today, start = default_time_window()
    assert _intent(start, today) is not None


def test_the_planning_instruction_states_the_default_explicitly() -> None:
    """Both providers read this same instruction, so the rule cannot diverge.

    The dates are asserted by VALUE, so an instruction that quietly drifted to a
    different window - or stopped naming one, leaving each model to invent its
    own - fails here.
    """

    today, start = default_time_window()
    instruction = _system_instruction()

    assert start.isoformat() in instruction
    assert today.isoformat() in instruction
    # And it must be labelled a default rather than presented as the user's ask.
    assert "default" in instruction.lower()
    assert "Never apply this default over explicit dates." in instruction


def test_both_providers_are_given_the_identical_instruction() -> None:
    """Gemini and NVIDIA must not be able to mean different things by a date.

    They share one instruction function; this pins that they still do, since a
    provider-local copy is exactly how the two would drift apart.
    """

    from app.services.agent.providers import gemini, nvidia

    assert nvidia._system_instruction is _system_instruction
    assert gemini._system_instruction is _system_instruction
