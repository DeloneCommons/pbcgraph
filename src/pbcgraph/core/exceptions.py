"""pbcgraph exceptions."""


class PBCGraphError(Exception):
    """Base class for pbcgraph errors."""


class StaleComponentError(PBCGraphError):
    """Raised when a PeriodicComponent is used after its graph has changed."""
