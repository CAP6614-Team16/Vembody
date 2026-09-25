"""Exceptions used by the Vembody runtime."""


class VembodyError(Exception):
    """Base exception for expected Vembody errors."""


class TaskLoadError(VembodyError):
    """Raised when a task file is missing or invalid."""


class ModelParseError(VembodyError):
    """Raised when model output is not valid action JSON."""


class ActionValidationError(VembodyError):
    """Raised when an action violates the configured safety policy."""


class ExecutionError(VembodyError):
    """Raised when an executor cannot perform an action."""
