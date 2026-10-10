from enum import StrEnum


class EvaluationStatus(StrEnum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"
