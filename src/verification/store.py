"""Independent original-walk references. The only read operation takes one claim."""
import json
import os
import sqlite3
from pathlib import Path

from .contract import identifier, normalize


class EnrollmentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError('Enrollment store cannot be a symlink')
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(str(self.path), 0o600)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reference ('
                       'subject TEXT NOT NULL, take_id TEXT NOT NULL UNIQUE, '
                       'source_hash TEXT NOT NULL UNIQUE, recipe TEXT NOT NULL, '
                       'embedding TEXT NOT NULL, role TEXT NOT NULL)')

    def connect(self):
        return sqlite3.connect(str(self.path))

    def add(self, subject, take_id, source_hash, recipe, embedding, role):
        identifier(subject)
        identifier(take_id)
        if role not in ('development', 'calibration', 'operational'):
            raise ValueError('Protected or invalid role')
        vector = normalize(embedding)
        try:
            with self.connect() as db:
                db.execute('INSERT INTO reference VALUES (?, ?, ?, ?, ?, ?)',
                           (subject, take_id, source_hash, recipe,
                            json.dumps(vector.tolist(), allow_nan=False), role))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Original already enrolled; windows/copies are not new references') from exc

    def claim(self, subject, recipe):
        identifier(subject)
        with self.connect() as db:
            rows = db.execute('SELECT take_id, source_hash, embedding, role FROM reference '
                              'WHERE subject = ? AND recipe = ? ORDER BY take_id',
                              (subject, recipe)).fetchall()
        return [dict(take_id=r[0], source_hash=r[1], embedding=normalize(json.loads(r[2])),
                     role=r[3]) for r in rows]

    def has_claim(self, subject):
        identifier(subject)
        with self.connect() as db:
            return db.execute('SELECT 1 FROM reference WHERE subject = ? LIMIT 1',
                              (subject,)).fetchone() is not None
