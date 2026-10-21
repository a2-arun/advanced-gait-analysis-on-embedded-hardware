"""Read-only schema/readiness status and explicit initialization of the verification store."""
import sqlite3
from pathlib import Path

from .store import KIND, SCHEMA_VERSION, EnrollmentStore


def database_status(path):
    """Counts and schema state only: no subject IDs or vectors, no creation, no migration."""
    path = Path(path)
    status = dict(path=str(path), state='ABSENT', schema_version=None, expected_schema=SCHEMA_VERSION,
                  integrity=None, references=0, subjects=0, references_by_recipe={},
                  inert_legacy_rows=0)
    if not path.is_file():
        return status
    try:
        db = sqlite3.connect('file:{}?mode=ro'.format(path), uri=True)
        try:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            status.update(schema_version=version, integrity=db.execute(
                'PRAGMA integrity_check').fetchone()[0])
            has_table = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                                   "AND name = 'reference'").fetchone()
            if version > SCHEMA_VERSION:
                status['state'] = 'NEWER_THAN_SOFTWARE'
            elif version < SCHEMA_VERSION:
                status['state'] = 'NEEDS_MIGRATION' if has_table else 'UNINITIALIZED'
            elif status['integrity'] != 'ok' or not has_table:
                status['state'] = 'CORRUPT'
            else:
                status['state'] = 'READY'
                status['references'], status['subjects'] = db.execute(
                    'SELECT COUNT(*), COUNT(DISTINCT subject) FROM reference').fetchone()
                status['references_by_recipe'] = dict(db.execute(
                    'SELECT recipe, COUNT(*) FROM reference WHERE kind = ? GROUP BY recipe', (KIND,)))
                status['inert_legacy_rows'] = db.execute(
                    'SELECT COUNT(*) FROM reference WHERE kind IS NULL').fetchone()[0]
        finally:
            db.close()
    except sqlite3.DatabaseError as exc:
        status.update(state='CORRUPT', integrity=str(exc))
    return status


def initialize_database(path):
    """Create or migrate to the current schema (never inserts rows), then report its status."""
    EnrollmentStore(path)
    return database_status(path)
