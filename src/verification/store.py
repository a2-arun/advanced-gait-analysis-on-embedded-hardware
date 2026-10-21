"""Independent original-walk references. The only read operation takes one claim."""
import json
import os
import sqlite3
from pathlib import Path

from .contract import identifier, normalize

SCHEMA_VERSION = 2
# Rows written by this software carry the marker; migrated pre-contract rows have NULL and
# are never scored, because their combined recipe digest cannot be shown compatible.
KIND = 'nano-reference-v2'
TABLE = ('CREATE TABLE reference ('
         'subject TEXT NOT NULL, take_id TEXT NOT NULL, source_hash TEXT NOT NULL, '
         'recipe TEXT NOT NULL, embedding TEXT NOT NULL, role TEXT NOT NULL, '
         'kind TEXT, runtime TEXT, UNIQUE (take_id, recipe), UNIQUE (source_hash, recipe))')


class ClosingConnection(sqlite3.Connection):
    """`with` commits or rolls back and then closes, so no connection outlives its operation."""

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


class EnrollmentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError('Enrollment store cannot be a symlink')
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(str(self.path), 0o600)
        self.migrate()

    def connect(self):
        return sqlite3.connect(str(self.path), timeout=10, factory=ClosingConnection)

    def migrate(self):
        """v0/v1 -> v2 in one transaction. The original table is kept as reference_v1."""
        db = sqlite3.connect(str(self.path), isolation_level=None)
        try:
            db.execute('BEGIN IMMEDIATE')
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError('Enrollment database is newer than this software')
            if version < SCHEMA_VERSION:
                legacy = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                                    "AND name = 'reference'").fetchone()
                if legacy:
                    db.execute('ALTER TABLE reference RENAME TO reference_v1')
                db.execute(TABLE)
                if legacy:
                    db.execute('INSERT INTO reference (subject, take_id, source_hash, recipe, '
                               'embedding, role) SELECT subject, take_id, source_hash, recipe, '
                               'embedding, role FROM reference_v1')
                db.execute('PRAGMA user_version = %d' % SCHEMA_VERSION)
            db.execute('COMMIT')
        except BaseException:
            if db.in_transaction:
                db.execute('ROLLBACK')
            raise
        finally:
            db.close()

    def add(self, subject, take_id, source_hash, recipe, embedding, role, runtime=None):
        identifier(subject)
        identifier(take_id)
        if role not in ('development', 'calibration', 'operational'):
            raise ValueError('Protected or invalid role')
        vector = normalize(embedding)
        try:
            with self.connect() as db:
                # One original belongs to one subject; it may hold one reference per recipe.
                for owner, take, digest in db.execute(
                        'SELECT subject, take_id, source_hash FROM reference '
                        'WHERE take_id = ? OR source_hash = ?', (take_id, source_hash)):
                    if (owner, take, digest) != (subject, take_id, source_hash):
                        raise ValueError('Original already enrolled for another subject/take')
                db.execute('INSERT INTO reference VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                           (subject, take_id, source_hash, recipe,
                            json.dumps(vector.tolist(), allow_nan=False), role, KIND, runtime))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Original already enrolled; windows/copies are not new references') from exc

    def claim(self, subject, recipe):
        identifier(subject)
        with self.connect() as db:
            rows = db.execute('SELECT take_id, source_hash, embedding, role FROM reference '
                              'WHERE subject = ? AND recipe = ? AND kind = ? ORDER BY take_id',
                              (subject, recipe, KIND)).fetchall()
        return [dict(take_id=r[0], source_hash=r[1], embedding=normalize(json.loads(r[2])),
                     role=r[3]) for r in rows]

    def has_claim(self, subject):
        identifier(subject)
        with self.connect() as db:
            return db.execute('SELECT 1 FROM reference WHERE subject = ? LIMIT 1',
                              (subject,)).fetchone() is not None

    def incompatible_count(self, subject, recipe):
        """Claim references unusable under this recipe (other recipe or migrated legacy row)."""
        identifier(subject)
        with self.connect() as db:
            return db.execute('SELECT COUNT(*) FROM reference WHERE subject = ? AND '
                              "NOT (recipe = ? AND COALESCE(kind, '') = ?)", (subject, recipe, KIND)).fetchone()[0]
