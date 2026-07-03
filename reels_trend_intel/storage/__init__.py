from __future__ import annotations

from reels_trend_intel.config.settings import Settings
from reels_trend_intel.storage.base import StorageBackend


def make_storage(settings: Settings) -> StorageBackend:
    """Construct the configured StorageBackend (swappable)."""
    if settings.storage.backend == "postgres":
        from reels_trend_intel.storage.postgres_backend import PostgresBackend

        dsn = settings.storage.postgres_dsn
        if dsn is None:
            raise ValueError("storage.backend=postgres requires storage.postgres_dsn")
        return PostgresBackend(
            dsn.get_secret_value(),
            min_size=settings.storage.pool_min_size,
            max_size=settings.storage.pool_max_size,
        )
    from reels_trend_intel.storage.sqlite_backend import SQLiteBackend

    return SQLiteBackend(settings.storage.sqlite_path)


__all__ = ["StorageBackend", "make_storage"]
