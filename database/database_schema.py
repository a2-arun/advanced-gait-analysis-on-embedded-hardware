"""
SQLite Database Schema for Gait-Based Edge Identification

Tables:
1. enrolled_identities — Enrolled person records
2. gait_embeddings — Stored gait embeddings for each enrolled person
3. identification_events — Log of all identifications performed
4. system_metadata — System configuration and metadata
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

# SQL Schema Definition
SCHEMA = """
-- ============================================================================
-- Enrolled Identities Table
-- ============================================================================
-- Stores information about enrolled persons
CREATE TABLE IF NOT EXISTS enrolled_identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL UNIQUE,
    person_name TEXT NOT NULL,
    enrollment_date DATETIME NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    metadata TEXT,  -- JSON string for additional fields
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Gait Embeddings Table
-- ============================================================================
-- Stores each enrolled person's reference gait signature: a RAW
-- (sequence_length, 78) feature sequence (float32, row-major bytes), NOT a
-- compact embedding. The trained model's only validated matching mechanism
-- (the difference-based verification head, see docs/ARCHITECTURE.md #4)
-- compares raw feature sequences directly - there is no cosine-similarity
-- embedding path for this checkpoint. Shape is fixed by config.yaml's
-- model.sequence_length / model.input_dim at read time.
CREATE TABLE IF NOT EXISTS gait_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrolled_id INTEGER NOT NULL,
    embedding_vector BLOB NOT NULL,  -- Serialized (sequence_length, 78) float32 array
    embedding_source TEXT,            -- 'enrollment', 'averaging', 'online'
    quality_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (enrolled_id) REFERENCES enrolled_identities(id)
        ON DELETE CASCADE
);

-- ============================================================================
-- Identification Events Table
-- ============================================================================
-- Logs every identification attempt for analytics/debugging
CREATE TABLE IF NOT EXISTS identification_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Query information
    query_duration_ms REAL,         -- Time to extract gait from camera
    
    -- Results
    identified_person_id INTEGER,   -- NULL if UNKNOWN
    identified_person_name TEXT,    -- NULL if UNKNOWN
    status TEXT NOT NULL,           -- 'IDENTIFIED', 'UNKNOWN', 'ERROR'
    
    -- Similarity scoring
    top_similarity_score REAL,      -- Best match similarity
    top_1_person_name TEXT,         -- Top 1 candidate
    top_1_similarity REAL,          -- Top 1 similarity
    top_2_person_name TEXT,         -- Top 2 candidate
    top_2_similarity REAL,          -- Top 2 similarity
    top_3_person_name TEXT,         -- Top 3 candidate
    top_3_similarity REAL,          -- Top 3 similarity
    
    -- Confidence
    confidence REAL,                -- Confidence score (0-1)
    threshold_used REAL,            -- Similarity threshold used
    
    -- Diagnostics
    num_enrolled_identities INTEGER,
    processing_time_ms REAL,        -- Total time for identification
    pose_quality_score REAL,        -- Quality of extracted pose
    error_message TEXT,             -- If status='ERROR'
    
    -- Session info
    device_id TEXT,                 -- Which device (laptop/jetson)
    camera_device_id INTEGER        -- Which camera input
);

-- ============================================================================
-- System Metadata Table
-- ============================================================================
-- Configuration and system state
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indices for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_enrolled_person_id 
    ON enrolled_identities(person_id);
CREATE INDEX IF NOT EXISTS idx_enrolled_name 
    ON enrolled_identities(person_name);
CREATE INDEX IF NOT EXISTS idx_gait_embeddings_enrolled_id 
    ON gait_embeddings(enrolled_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp 
    ON identification_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_status 
    ON identification_events(status);
CREATE INDEX IF NOT EXISTS idx_events_person_id 
    ON identification_events(identified_person_id);

-- ============================================================================
-- Initial System Metadata
-- ============================================================================
INSERT OR IGNORE INTO system_metadata (key, value, description) VALUES
    ('db_version', '1.0', 'Database schema version'),
    ('model_version', 'full_hybrid_best', 'Checkpoint variant in use'),
    ('similarity_threshold', '0.7737', 'Operating threshold (Youdens J, from LOOCV verification, see docs/ARCHITECTURE.md)'),
    ('created_at', CURRENT_TIMESTAMP, 'Database creation timestamp');
"""


def init_database(db_path: str = "database/gait.db") -> sqlite3.Connection:
    """
    Initialize SQLite database with schema.
    
    Args:
        db_path: Path to database file
        
    Returns:
        sqlite3.Connection object
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Create schema
    conn.executescript(SCHEMA)
    conn.commit()
    
    print(f"[OK] Database initialized: {db_path}")
    return conn


def get_connection(db_path: str = "database/gait.db") -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    # Test: Initialize database
    db = init_database()
    
    # Verify tables exist
    cursor = db.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = cursor.fetchall()
    
    print("\nCreated tables:")
    for table in tables:
        print(f"  - {table[0]}")
    
    db.close()
    print("\n[OK] Database ready for use")
