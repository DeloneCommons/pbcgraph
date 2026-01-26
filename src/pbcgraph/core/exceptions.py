"""pbcgraph exceptions."""


class PBCGraphError(Exception):
    """Base class for pbcgraph errors."""


class StaleComponentError(PBCGraphError):
    """Raised when a PeriodicComponent is used after its graph has changed."""


class LiftPatchError(PBCGraphError):
    """Raised when finite patch extraction fails."""


class CanonicalLiftError(PBCGraphError):
    """Raised when canonical lift construction fails."""
