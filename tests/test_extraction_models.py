"""Unit tests for the Pydantic extraction schemas (no LLM or network used)."""
import json

import pytest
from pydantic import ValidationError

from schemas import (
    ActionItem,
    ActionStatus,
    ExecutiveSummary,
    KeyDecision,
    MeetingExtraction,
    RiskFlag,
    Severity,
    TopicCluster,
)


def make_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        overview="Team reviewed the Q3 launch plan.",
        main_points=["Launch date", "Budget"],
        outcomes=["Launch confirmed"],
    )


def test_valid_full_extraction():
    data = MeetingExtraction(
        summary=make_summary(),
        decisions=[
            KeyDecision(
                decision="Launch on 1 October",
                speaker="Asha",
                context="Testing finished early.",
            )
        ],
        action_items=[
            ActionItem(
                task="Prepare release notes",
                owner="Ravi",
                deadline="Friday",
                context="Needed for launch.",
                status="in_progress",
            )
        ],
        risks=[
            RiskFlag(
                description="Vendor may be late",
                severity="high",
                context="Vendor missed last deadline.",
                potential_impact="Launch delay",
            )
        ],
        topics=[
            TopicCluster(
                name="Launch", description="Launch timing", related_points=["Date"]
            )
        ],
    )
    assert data.summary.overview.startswith("Team")
    assert data.action_items[0].status == ActionStatus.IN_PROGRESS
    assert data.risks[0].severity == Severity.HIGH


def test_action_item_missing_owner_is_none():
    item = ActionItem(task="Send agenda", deadline="Monday")
    assert item.owner is None
    assert item.deadline == "Monday"


def test_action_item_missing_deadline_is_none():
    item = ActionItem(task="Send agenda", owner="Maya")
    assert item.deadline is None
    assert item.owner == "Maya"


def test_blank_optional_strings_become_none():
    item = ActionItem(task="Send agenda", owner="  ", deadline="")
    assert item.owner is None
    assert item.deadline is None
    assert item.status == ActionStatus.UNKNOWN


def test_multiple_decisions():
    data = MeetingExtraction(
        summary=make_summary(),
        decisions=[
            KeyDecision(decision="Use Gemini", speaker="Asha"),
            KeyDecision(decision="Weekly demos"),
            KeyDecision(decision="Freeze scope", speaker="Ravi"),
        ],
    )
    assert len(data.decisions) == 3
    assert data.decisions[1].speaker is None


def test_multiple_action_items():
    data = MeetingExtraction(
        summary=make_summary(),
        action_items=[ActionItem(task=f"Task {i}") for i in range(5)],
    )
    assert len(data.action_items) == 5
    assert all(a.owner is None and a.deadline is None for a in data.action_items)


@pytest.mark.parametrize("value", ["low", "medium", "high", "HIGH", " Medium "])
def test_valid_risk_severity(value):
    risk = RiskFlag(description="Some risk", severity=value)
    assert risk.severity in set(Severity)


def test_invalid_risk_severity_rejected():
    with pytest.raises(ValidationError):
        RiskFlag(description="Some risk", severity="critical")


def test_missing_risk_severity_rejected():
    with pytest.raises(ValidationError):
        RiskFlag(description="Some risk")


def test_empty_collections_are_valid():
    data = MeetingExtraction(summary=make_summary())
    assert data.decisions == []
    assert data.action_items == []
    assert data.risks == []
    assert data.topics == []


def test_json_serialization_round_trip():
    original = MeetingExtraction(
        summary=make_summary(),
        action_items=[ActionItem(task="Send agenda", status="completed")],
        risks=[RiskFlag(description="Delay", severity="low")],
    )
    as_dict = original.to_json_dict()
    assert as_dict["risks"][0]["severity"] == "low"  # plain string, not Enum
    assert as_dict["action_items"][0]["owner"] is None

    text = original.to_json()
    parsed = json.loads(text)  # must be valid JSON
    assert MeetingExtraction.model_validate(parsed) == original


def test_invalid_data_types_rejected():
    with pytest.raises(ValidationError):
        MeetingExtraction(summary=make_summary(), decisions="not a list")
    with pytest.raises(ValidationError):
        ActionItem(task=123)
    with pytest.raises(ValidationError):
        ExecutiveSummary(overview="ok", main_points="not a list")
    with pytest.raises(ValidationError):
        MeetingExtraction(summary="just text")


def test_empty_required_text_rejected():
    with pytest.raises(ValidationError):
        KeyDecision(decision="   ")
    with pytest.raises(ValidationError):
        ExecutiveSummary(overview="")


def test_invalid_action_status_rejected():
    with pytest.raises(ValidationError):
        ActionItem(task="x", status="finished-ish")
