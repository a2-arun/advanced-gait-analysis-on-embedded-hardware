"""Managed raw recordings; caller-declared derivatives and locked-final files are refused."""
import json
import os
from datetime import datetime, timezone
import uuid
from pathlib import Path

from .contract import identifier, sha256_file


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Exclusive temporary file; refuse linked destinations. Single-writer operation.
    if path.is_symlink():
        raise ValueError('Linked artifact path refused')
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_datetime(value):
    # Python 3.6 strptime requires numeric UTC offsets without a colon.
    compact = value[:-3] + value[-2:] if len(value) >= 6 and value[-3] == ':' else value
    return datetime.strptime(compact, '%Y-%m-%dT%H:%M:%S%z')


class RecordingRegistry:
    def __init__(self, path):
        self.path = Path(path)

    def rows(self):
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text())
        if value['schema'] not in ('nano-raw-recordings-v1', 'nano-recordings-v2'):
            raise ValueError('Unsupported registry schema')
        rows = value['recordings']
        takes, hashes, roles = set(), set(), {}
        for row in rows:
            purpose = row.get('purpose', 'study')
            if purpose not in ('study', 'enrollment', 'query'):
                raise ValueError('Unknown recording purpose')
            if purpose == 'query':
                if row['subject_id'] is not None:
                    raise ValueError('Operational queries must not assert the walker identity')
            else:
                identifier(row['subject_id'])
            identifier(row['original_take_id'])
            if purpose == 'study' or row['captured_at'] is not None:
                validate_datetime(row['captured_at'])
            if purpose != 'study':
                validate_datetime(row['registered_at'])
            allowed_roles = ('development', 'calibration') if purpose == 'study' else ('operational',)
            if (row['variant'] != 'raw' or row['role'] not in allowed_roles
                    or row['original_take_id'] in takes or row['sha256'] in hashes
                    or (row['subject_id'] is not None
                        and roles.get(row['subject_id'], row['role']) != row['role'])
                    or not row['consent_ref']):
                raise ValueError('Registry lineage/ownership/role conflict')
            takes.add(row['original_take_id'])
            hashes.add(row['sha256'])
            if row['subject_id'] is not None:
                roles[row['subject_id']] = row['role']
        return rows

    def register(self, path, take, subject, session, view, role, consent, captured_at):
        identifier(take)
        identifier(subject)
        identifier(session)
        validate_datetime(captured_at)
        if role not in ('development', 'calibration') or not consent:
            raise ValueError('Only consent-backed development/calibration originals are allowed')
        if view not in ('left45', 'right45', 'toward', 'away'):
            raise ValueError('Use an explicit study view')
        row = dict(original_take_id=take, subject_id=subject, session_id=session,
                   view=view, role=role, consent_ref=consent, captured_at=captured_at,
                   variant='raw')
        self._append(path, row)

    def register_operational(self, path, subject, consent, purpose, captured_at=None):
        if purpose not in ('enrollment', 'query') or not consent:
            raise ValueError('Operational recording requires purpose and consent')
        if purpose == 'enrollment':
            identifier(subject)
        elif subject is not None:
            raise ValueError('Query walker identity must remain unset')
        if captured_at is not None:
            validate_datetime(captured_at)
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        row = dict(original_take_id='walk_' + uuid.uuid4().hex, subject_id=subject,
                   session_id=None, view='unspecified', role='operational', purpose=purpose,
                   consent_ref=consent, captured_at=captured_at, registered_at=now, variant='raw')
        self._append(path, row)
        return row

    def _append(self, path, row):
        path = Path(path).resolve(strict=True)
        if path == self.path.resolve():
            raise ValueError('Registry cannot overwrite a source')
        rows = self.rows()
        checksum = sha256_file(path)
        for previous in rows:
            if (previous['original_take_id'] == row['original_take_id']
                    or previous['sha256'] == checksum or Path(previous['path']).resolve() == path):
                raise ValueError('Original take/bytes already registered')
            if (row['subject_id'] is not None and previous['subject_id'] == row['subject_id']
                    and previous['role'] != row['role']):
                raise ValueError('Subject roles must remain disjoint')
        row.update(path=str(path), sha256=checksum)
        rows.append(row)
        private_json(self.path, dict(schema='nano-recordings-v2', recordings=rows))

    def resolve(self, path):
        path = Path(path).resolve(strict=True)
        checksum = sha256_file(path)
        for row in self.rows():
            if row['sha256'] == checksum:
                if sha256_file(row['path']) != checksum:
                    raise ValueError('Registered original changed')
                return row
        raise ValueError('Unregistered input; register the immutable original first')

    def validate_reference(self, reference, subject):
        for row in self.rows():
            if row['original_take_id'] == reference['take_id']:
                if (row['subject_id'] != subject or row['sha256'] != reference['source_hash']
                        or row['role'] != reference['role']
                        or sha256_file(row['path']) != row['sha256']):
                    raise ValueError('Reference source/ownership changed')
                return row
        raise ValueError('Reference is missing from registry')
