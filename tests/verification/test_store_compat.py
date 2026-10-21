import json
import sqlite3
import numpy as np
import pytest

from src.verification.contract import fingerprint
from src.verification.policy import decide
from src.verification.store import EnrollmentStore, KIND


def legacy_db(path, recipe='combined-digest'):
    with sqlite3.connect(str(path)) as db:
        db.execute('CREATE TABLE reference (subject TEXT NOT NULL, take_id TEXT NOT NULL UNIQUE, '
                   'source_hash TEXT NOT NULL UNIQUE, recipe TEXT NOT NULL, '
                   'embedding TEXT NOT NULL, role TEXT NOT NULL)')
        db.execute('INSERT INTO reference VALUES (?, ?, ?, ?, ?, ?)',
                   ('S001', 'take1', 'a' * 64, recipe, json.dumps([1.0, 0.0]), 'operational'))


def test_legacy_rows_migrate_losslessly_and_are_never_scored(tmp_path):
    legacy_db(tmp_path / 'e.db')
    store = EnrollmentStore(tmp_path / 'e.db')
    with sqlite3.connect(str(tmp_path / 'e.db')) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2
        assert db.execute('SELECT * FROM reference_v1').fetchall() == [
            ('S001', 'take1', 'a' * 64, 'combined-digest', '[1.0, 0.0]', 'operational')]
    assert store.has_claim('S001')
    assert store.claim('S001', 'combined-digest') == []  # same string, still not trusted
    assert store.incompatible_count('S001', 'combined-digest') == 1
    EnrollmentStore(tmp_path / 'e.db')  # idempotent reopen


def test_failed_migration_rolls_back_without_touching_data(tmp_path):
    with sqlite3.connect(str(tmp_path / 'e.db')) as db:
        db.execute('CREATE TABLE reference (subject TEXT, recipe TEXT)')
        db.execute("INSERT INTO reference VALUES ('S001', 'r')")
    with pytest.raises(sqlite3.Error):
        EnrollmentStore(tmp_path / 'e.db')
    with sqlite3.connect(str(tmp_path / 'e.db')) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 0
        assert db.execute('SELECT * FROM reference').fetchall() == [('S001', 'r')]


def test_database_from_a_newer_version_is_refused(tmp_path):
    with sqlite3.connect(str(tmp_path / 'e.db')) as db:
        db.execute('PRAGMA user_version = 99')
    with pytest.raises(ValueError):
        EnrollmentStore(tmp_path / 'e.db')


def test_same_original_can_be_reprocessed_under_a_new_recipe_for_its_owner_only(tmp_path):
    legacy_db(tmp_path / 'e.db')
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 'take1', 'a' * 64, 'new-recipe', [0., 1.], 'operational', 'rt1')
    assert [r['take_id'] for r in store.claim('S001', 'new-recipe')] == ['take1']
    np.testing.assert_array_equal(store.claim('S001', 'new-recipe')[0]['embedding'], [0., 1.])
    with sqlite3.connect(str(tmp_path / 'e.db')) as db:  # old vector not rewritten
        assert db.execute("SELECT embedding FROM reference WHERE kind IS NULL").fetchone()[0] \
            == '[1.0, 0.0]'
        assert db.execute("SELECT runtime, kind FROM reference WHERE kind IS NOT NULL").fetchone() \
            == ('rt1', KIND)
    with pytest.raises(ValueError):
        store.add('S001', 'take1', 'a' * 64, 'new-recipe', [0., 1.], 'operational')
    for other in ('new-recipe', 'third-recipe'):
        with pytest.raises(ValueError):
            store.add('S002', 'take1', 'a' * 64, other, [0., 1.], 'operational')
        with pytest.raises(ValueError):
            store.add('S002', 'take9', 'a' * 64, other, [0., 1.], 'operational')
    with pytest.raises(ValueError):
        store.add('S001', 'take1', 'b' * 64, 'third-recipe', [0., 1.], 'operational')


def test_independent_references_for_one_recipe_remain_supported(tmp_path):
    store = EnrollmentStore(tmp_path / 'e.db')
    for i in range(3):
        store.add('S001', 'take%d' % i, str(i) * 64, 'r', [1., float(i)], 'operational')
    store.add('S002', 'other', 'f' * 64, 'r', [0., 1.], 'operational')
    assert [x['take_id'] for x in store.claim('S001', 'r')] == ['take0', 'take1', 'take2']
    assert store.incompatible_count('S001', 'r') == 0


def test_every_embedding_read_is_scoped_to_the_supplied_claim(tmp_path):
    class Traced(EnrollmentStore):
        statements = []
        def connect(self):
            db = super(Traced, self).connect()
            db.set_trace_callback(self.statements.append)
            return db
    store = Traced(tmp_path / 'e.db')
    store.add('S001', 'take1', 'a' * 64, 'r', [1., 0.], 'operational')
    del Traced.statements[:]
    store.claim('S001', 'r')
    store.has_claim('S001')
    store.incompatible_count('S001', 'r')
    reads = [s for s in Traced.statements if s.lstrip().upper().startswith('SELECT')]
    assert reads and all("subject = 'S001'" in s for s in reads)


def test_policy_for_another_recipe_never_yields_a_boolean(tmp_path):
    evidence = dict(schema='nano-calibration-evidence-v1', recipe='recipe-a', split_role='calibration',
                    threshold=.5, reference_count=1, validation_status='APPROVED',
                    trial_manifest_sha256='a' * 64, split_sha256='b' * 64, report_sha256='c' * 64,
                    approved_by='synthetic-fixture', approved_at='2026-10-20T12:00:00+00:00')
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence))
    (tmp_path / 'policy.json').write_text(json.dumps(dict(
        schema='nano-policy-v1', recipe='recipe-a', threshold=.5, reference_count=1,
        evidence='evidence.json', evidence_sha256=fingerprint(evidence))))
    assert decide(.9, 'recipe-a', tmp_path / 'policy.json', 1) == ('OK', True)
    assert decide(.9, 'recipe-b', tmp_path / 'policy.json', 1) == ('POLICY_NOT_READY', None)
    assert decide(.9, 'recipe-a', tmp_path / 'policy.json', 2) == ('POLICY_NOT_READY', None)
    assert decide(.9, 'recipe-a', None, 1) == ('POLICY_NOT_READY', None)
