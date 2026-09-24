"""CRUD access to the enrolled-identity gallery and identification log.

Wraps database/database_schema.py's raw SQLite schema with the operations
the enrollment and identification pipelines actually need. Gait signatures
are stored and returned as 1-D float32 embedding vectors (GaitModel.embed).
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from database.database_schema import init_database
from src.utils.config import load_config, resolve_path


def _serialize(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float32).tobytes()


def _deserialize(blob: bytes) -> np.ndarray:
    # np.frombuffer's array is a read-only view over `blob` - copy it so
    # downstream code can treat it as an owned array.
    return np.frombuffer(blob, dtype=np.float32).copy()


class GaitDatabase:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        db_path = resolve_path(self.config["database"]["path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config["database"].get("auto_init", True) or not db_path.exists():
            init_database(str(db_path)).close()

        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # -- Enrollment -----------------------------------------------------

    def enroll_identity(
        self,
        person_name: str,
        signature: np.ndarray,
        num_samples: int,
        quality_score: float,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new enrolled identity (or a fresh signature for an
        existing one, keeping history). Returns the person_id."""
        cur = self.conn.cursor()
        existing = cur.execute(
            "SELECT id, person_id FROM enrolled_identities WHERE person_name = ?",
            (person_name,),
        ).fetchone()

        now = datetime.utcnow().isoformat()

        if existing:
            enrolled_row_id, person_id = existing["id"], existing["person_id"]
            cur.execute(
                """UPDATE enrolled_identities
                   SET num_samples = ?, quality_score = ?, updated_at = ?, metadata = ?
                   WHERE id = ?""",
                (num_samples, quality_score, now, json.dumps(metadata or {}), enrolled_row_id),
            )
        else:
            person_id = str(uuid.uuid4())
            cur.execute(
                """INSERT INTO enrolled_identities
                   (person_id, person_name, enrollment_date, num_samples, quality_score, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, person_name, now, num_samples, quality_score, json.dumps(metadata or {})),
            )
            enrolled_row_id = cur.lastrowid

        cur.execute(
            """INSERT INTO gait_embeddings (enrolled_id, embedding_vector, embedding_source, quality_score)
               VALUES (?, ?, ?, ?)""",
            (enrolled_row_id, _serialize(signature), "enrollment", quality_score),
        )
        self.conn.commit()
        return person_id

    def delete_identity(self, person_name: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM enrolled_identities WHERE person_name = ?", (person_name,))
        self.conn.commit()
        return cur.rowcount > 0

    def list_identities(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT person_id, person_name, enrollment_date, num_samples, quality_score "
            "FROM enrolled_identities ORDER BY person_name"
        ).fetchall()

    # -- Matching gallery -------------------------------------------------

    def get_gallery(self) -> Dict[str, np.ndarray]:
        """Latest signature per enrolled identity, keyed by person_name."""
        rows = self.conn.execute(
            """SELECT ei.person_name, ge.embedding_vector
               FROM enrolled_identities ei
               JOIN gait_embeddings ge ON ge.enrolled_id = ei.id
               WHERE ge.id = (
                   SELECT id FROM gait_embeddings
                   WHERE enrolled_id = ei.id
                   ORDER BY created_at DESC LIMIT 1
               )"""
        ).fetchall()

        return {
            row["person_name"]: _deserialize(row["embedding_vector"])
            for row in rows
        }

    # -- Identification event log -----------------------------------------

    def log_identification_event(
        self,
        status: str,
        identified_person_name: Optional[str],
        top_k: List[tuple],
        threshold_used: float,
        processing_time_ms: float,
        pose_quality_score: float,
        num_enrolled_identities: int,
        device_id: str = "laptop",
        camera_device_id: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        identified_row = None
        if identified_person_name:
            identified_row = self.conn.execute(
                "SELECT id FROM enrolled_identities WHERE person_name = ?",
                (identified_person_name,),
            ).fetchone()

        top = top_k + [(None, None)] * (3 - len(top_k))  # pad to 3

        self.conn.execute(
            """INSERT INTO identification_events (
                identified_person_id, identified_person_name, status,
                top_similarity_score,
                top_1_person_name, top_1_similarity,
                top_2_person_name, top_2_similarity,
                top_3_person_name, top_3_similarity,
                threshold_used, num_enrolled_identities,
                processing_time_ms, pose_quality_score,
                error_message, device_id, camera_device_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                identified_row["id"] if identified_row else None,
                identified_person_name,
                status,
                top[0][1],
                top[0][0], top[0][1],
                top[1][0], top[1][1],
                top[2][0], top[2][1],
                threshold_used, num_enrolled_identities,
                processing_time_ms, pose_quality_score,
                error_message, device_id, camera_device_id,
            ),
        )
        self.conn.commit()
