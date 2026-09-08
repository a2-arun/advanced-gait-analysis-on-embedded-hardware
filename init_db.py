#!/usr/bin/env python
"""Initialize the gait identification database."""

from database.database_schema import init_database
import sqlite3
import os

# Initialize database  
conn = init_database()
print(f'[OK] Database connection type: {type(conn).__name__}')

if isinstance(conn, sqlite3.Connection):
    print('[OK] Database connection established')

    # Query tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f'[OK] Created tables: {[t[0] for t in tables]}')

    # Check database file
    db_path = 'database/gait.db'
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f'[OK] Database file: {db_path}')
        print(f'[OK] File size: {size} bytes')

    conn.close()
    print('\n[OK] Database initialized and schema created')
