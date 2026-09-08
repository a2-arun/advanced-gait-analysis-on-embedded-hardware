"""Database module for gait identification system."""

from .database_schema import init_database, get_connection

__all__ = ['init_database', 'get_connection']
