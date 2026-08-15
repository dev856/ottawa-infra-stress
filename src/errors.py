"""Project-specific exceptions with safe, predictable meanings.

Use these exception types at module boundaries so callers can distinguish bad
configuration, invalid source data, external-service failures, and unsafe model
artifacts without parsing error-message text.
"""


class ConfigError(RuntimeError):
    """Raised when environment configuration is missing, invalid, or unsafe."""


class ExternalServiceError(RuntimeError):
    """Raised when an approved external service cannot provide valid data."""


class DataValidationError(RuntimeError):
    """Raised when incoming data does not match the required schema or range."""


class ArtifactError(RuntimeError):
    """Raised when a local generated artifact is missing, invalid, or untrusted."""
