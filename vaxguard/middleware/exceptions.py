class VaxGuardError(Exception):
    """Base exception for all VaxGuard custom errors."""
    pass

class MiddlewareError(VaxGuardError):
    """Base exception for middleware routing failures."""
    pass

class CacheMissError(MiddlewareError):
    """Raised when a requested item is not in the fast-cache."""
    pass

class DatabaseConnectionError(VaxGuardError):
    """Raised when the ORM fails to connect."""
    pass
