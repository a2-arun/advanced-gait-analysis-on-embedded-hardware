import sqlite3

import numpy as np
import pytest

from src.verification.dbcheck import database_status, initialize_database
from src.verification.store import EnrollmentStore

VECTOR = np.array([3., 4.])  # deterministic synthetic vector, never a real biometric


def test_status_of_a_missing_database_does_not_create_it(tmp_path):
    status = database_status(tmp_path / 'data' / 'v2.db')
    assert status['state'] == 'ABSENT' and status['references'] == 0
    assert not (tmp_path / 'data').exists()


def test_initialize_creates_an_empty_ready_schema_v2_database_with_private_permissions(tmp_path):
    status = initialize_database(tmp_path / 'data' / 'v2.db')
    assert status['state'] == 'READY' and status['schema_version'] == 2
    assert status['references'] == status['subjects'] == 0 and status['integrity'] == 'ok'
    assert (tmp_path / 'data' / 'v2.db').stat().st_mode & 0o777 == 0o600
    assert initialize_database(tmp_path / 'data' / 'v2.db')['state'] == 'READY'  # idempotent


def test_status_reports_counts_without_exposing_subjects_or_vectors(tmp_path):
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 't1', 'a' * 64, 'recipe-A', VECTOR, 'operational')
    store.add('S001', 't2', 'b' * 64, 'recipe-A', VECTOR, 'operational')
    store.add('S002', 't3', 'c' * 64, 'recipe-B', VECTOR, 'operational')
    status = database_status(tmp_path / 'e.db')
    assert (status['references'], status['subjects']) == (3, 2)
    assert status['references_by_recipe'] == {'recipe-A': 2, 'recipe-B': 1}
    assert 'S001' not in str(status) and '3.0' not in str(status)


def test_status_distinguishes_legacy_newer_and_corrupt_databases(tmp_path):
    legacy = tmp_path / 'legacy.db'
    with sqlite3.connect(str(legacy)) as db:
        db.execute('CREATE TABLE reference (subject TEXT, take_id TEXT, source_hash TEXT, '
                   'recipe TEXT, embedding TEXT, role TEXT)')
    assert database_status(legacy)['state'] == 'NEEDS_MIGRATION'
    newer = tmp_path / 'newer.db'
    with sqlite3.connect(str(newer)) as db:
        db.execute('PRAGMA user_version = 9')
    assert database_status(newer)['state'] == 'NEWER_THAN_SOFTWARE'
    corrupt = tmp_path / 'corrupt.db'
    corrupt.write_bytes(b'this is not a sqlite database' * 10)
    assert database_status(corrupt)['state'] == 'CORRUPT'
    with pytest.raises(sqlite3.Error):
        initialize_database(corrupt)


def test_status_never_migrates_or_modifies_a_legacy_database(tmp_path):
    legacy = tmp_path / 'legacy.db'
    with sqlite3.connect(str(legacy)) as db:
        db.execute('CREATE TABLE reference (subject TEXT)')
    before = legacy.read_bytes()
    database_status(legacy)
    assert legacy.read_bytes() == before


def test_connections_are_closed_after_each_store_operation(tmp_path):
    store = EnrollmentStore(tmp_path / 'e.db')
    with store.connect() as db:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute('SELECT 1')


def test_failed_write_leaves_no_partial_row_and_a_working_store(tmp_path):
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 't1', 'a' * 64, 'r', VECTOR, 'operational')
    for bad in (lambda: store.add('../x', 't2', 'b' * 64, 'r', VECTOR, 'operational'),
                lambda: store.add('S001', 't2', 'b' * 64, 'r', [np.nan, 1], 'operational'),
                lambda: store.add('S001', 't2', 'b' * 64, 'r', VECTOR, 'locked-final'),
                lambda: store.add('S001', 't3', 'a' * 64, 'r', VECTOR, 'operational'),  # dup source
                lambda: store.add('S002', 't1', 'z' * 64, 'r', VECTOR, 'operational')):  # dup take
        with pytest.raises(ValueError):
            bad()
    assert len(store.claim('S001', 'r')) == 1 and not store.has_claim('S002')
    store.add('S001', 't2', 'b' * 64, 'r', VECTOR, 'operational')
    assert len(store.claim('S001', 'r')) == 2


def test_stale_model_references_are_rejected_but_reported(tmp_path):
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 't1', 'a' * 64, 'old-model-recipe', VECTOR, 'operational')
    assert store.claim('S001', 'new-model-recipe') == []
    assert store.has_claim('S001') and store.incompatible_count('S001', 'new-model-recipe') == 1
