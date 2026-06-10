import sqlite3
import os
import asyncio
from datetime import datetime

DB_PATH = os.getenv("SQLITE_DB_PATH", "voice_assistant.db")


def init_db():
    """Initializes the SQLite database and creates the logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS call_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       session_id
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       provider
                       TEXT
                       NOT
                       NULL,
                       language
                       TEXT
                       NOT
                       NULL,
                       start_time
                       TIMESTAMP
                       NOT
                       NULL,
                       end_time
                       TIMESTAMP,
                       duration_seconds
                       REAL,
                       status
                       TEXT
                       DEFAULT
                       'active'
                   )
                   """)
    # Add an index on start_time for quick time-series polling (frequency/duration charts)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON call_logs(start_time)")
    conn.commit()
    conn.close()


def db_start_call(session_id: str, provider: str, language: str):
    """Inserts a new call record when a WebSocket stream initializes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
                       INSERT INTO call_logs (session_id, provider, language, start_time, status)
                       VALUES (?, ?, ?, ?, 'active')
                       """, (session_id, provider.lower(), language.lower(), datetime.utcnow()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Handle potential duplicate WebSocket frames gracefully
    finally:
        conn.close()


def db_end_call(session_id: str, completed: bool = True):
    """Updates an existing call record with ending timestamps and durations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow()

    # First, fetch the start time to accurately compute metrics
    cursor.execute("SELECT start_time FROM call_logs WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()

    if row:
        start_time = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
        # Calculate strict analytical duration
        if isinstance(start_time, str):
            # SQLite sometimes stores text timestamps depending on adaptation adapters
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")

        duration = (now - start_time).total_seconds()
        status = "completed" if completed else "failed"

        cursor.execute("""
                       UPDATE call_logs
                       SET end_time         = ?,
                           duration_seconds = ?,
                           status           = ?
                       WHERE session_id = ?
                       """, (now, duration, status, session_id))
        conn.commit()
    conn.close()


# Async wrappers so your FastAPI WebSocket event loop never blocks on disk I/O
async def async_start_call(session_id: str, provider: str, language: str):
    await asyncio.to_thread(db_start_call, session_id, provider, language)


async def async_end_call(session_id: str, completed: bool = True):
    await asyncio.to_thread(db_end_call, session_id, completed)