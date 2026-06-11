import sqlite3
import os
import asyncio
from datetime import datetime

# 1. Resolve path with safety fallback checks
if os.path.exists("/data"):
    DB_PATH = "/data/voice_assistant.db"
else:
    DB_PATH = os.getenv("SQLITE_DB_PATH", "voice_assistant.db")

print(f"[Database Engine] Active storage path resolved to: {DB_PATH}")


def get_connection():
    """
    Returns a connection with built-in timestamp parsing.
    Using PARSE_DECLTYPES ensures SQLite text timestamps auto-convert to Python datetime objects.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=15.0,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )
    # Return rows as dict-like objects for easier processing down the line
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database and creates all persistent tables."""
    parent_dir = os.path.dirname(DB_PATH)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")  # Safe balance for performance over network/FUSE storage
    except sqlite3.OperationalError:
        pass

        # 1. call_logs — metadata per WebSocket session
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
                       caller_number
                       TEXT
                       DEFAULT
                       'UNKNOWN',
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time   ON call_logs(start_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_caller       ON call_logs(caller_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider     ON call_logs(provider)")

    # 2. transcript_logs — Optional/Recommended for voice agent pipelines
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS transcript_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       session_id
                       TEXT
                       NOT
                       NULL,
                       role
                       TEXT
                       NOT
                       NULL, -- 'user' (STT) or 'assistant' (TTS)
                       text
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       TIMESTAMP
                       NOT
                       NULL,
                       FOREIGN
                       KEY
                   (
                       session_id
                   ) REFERENCES call_logs
                   (
                       session_id
                   ) ON DELETE CASCADE
                       )
                   """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcript_session ON transcript_logs(session_id)")

    # 3. model_load_logs — performance tracking for cold starts
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS model_load_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       model_name
                       TEXT
                       NOT
                       NULL,
                       duration_s
                       REAL
                       NOT
                       NULL,
                       device
                       TEXT
                       DEFAULT
                       'cpu',
                       loaded_at
                       TIMESTAMP
                       NOT
                       NULL
                   )
                   """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loaded_at ON model_load_logs(loaded_at)")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Call record helpers
# ---------------------------------------------------------------------------

def db_start_call(session_id: str, provider: str, language: str, caller_number: str = "UNKNOWN"):
    """Inserts a new call record when a WebSocket stream initializes."""
    with get_connection() as conn:
        try:
            conn.execute("""
                         INSERT INTO call_logs
                             (session_id, provider, language, caller_number, start_time, status)
                         VALUES (?, ?, ?, ?, ?, 'active')
                         """, (session_id, provider.lower(), language.lower(), caller_number, datetime.utcnow()))
        except sqlite3.IntegrityError:
            pass
        except Exception as e:
            print(f"[database] Failed to record call start: {e}")


def db_end_call(session_id: str, completed: bool = True):
    """Stamps end_time, duration, and final status using native DB date extraction."""
    with get_connection() as conn:
        try:
            # Let's pull the raw row using row_factory
            row = conn.execute("SELECT start_time FROM call_logs WHERE session_id = ?", (session_id,)).fetchone()

            if row:
                start_time = row['start_time']
                now = datetime.utcnow()

                # If PARSE_DECLTYPES successfully returned a datetime object
                if isinstance(start_time, datetime):
                    duration = (now - start_time).total_seconds()
                else:
                    # Robust fallback parser if type detection fails
                    from datetime import timezone
                    cleaned_str = str(start_time).replace("Z", "")
                    if "." not in cleaned_str:
                        cleaned_str += ".000000"
                    start_time = datetime.fromisoformat(cleaned_str)
                    duration = (now - start_time).total_seconds()

                status = "completed" if completed else "failed"

                conn.execute("""
                             UPDATE call_logs
                             SET end_time         = ?,
                                 duration_seconds = ?,
                                 status           = ?
                             WHERE session_id = ?
                             """, (now, round(duration, 2), status, session_id))
        except Exception as e:
            print(f"[database] Failed to log call end execution: {e}")


def db_log_transcript(session_id: str, role: str, text: str):
    """Logs an individual turn (STT chunk or TTS response) during the call."""
    with get_connection() as conn:
        try:
            conn.execute("""
                         INSERT INTO transcript_logs (session_id, role, text, timestamp)
                         VALUES (?, ?, ?, ?)
                         """, (session_id, role.lower(), text, datetime.utcnow()))
        except Exception as e:
            print(f"[database] Failed to log transcript turn: {e}")


def db_log_model_load(model_name: str, duration_s: float, device: str = "cpu"):
    """Persists how long a specific model took to initialize at startup."""
    with get_connection() as conn:
        try:
            conn.execute("""
                         INSERT INTO model_load_logs (model_name, duration_s, device, loaded_at)
                         VALUES (?, ?, ?, ?)
                         """, (model_name, round(duration_s, 3), device.lower(), datetime.utcnow()))
        except Exception as e:
            print(f"[database] Failed to log model load: {e}")


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def async_start_call(session_id: str, provider: str, language: str, caller_number: str = "UNKNOWN"):
    await asyncio.to_thread(db_start_call, session_id, provider, language, caller_number)


async def async_end_call(session_id: str, completed: bool = True):
    await asyncio.to_thread(db_end_call, session_id, completed)


async def async_log_transcript(session_id: str, role: str, text: str):
    await asyncio.to_thread(db_log_transcript, session_id, role, text)


async def async_log_model_load(model_name: str, duration_s: float, device: str = "cpu"):
    await asyncio.to_thread(db_log_model_load, model_name, duration_s, device)


# Verify DB structure on script execution
try:
    init_db()
except Exception as init_err:
    print(f"[CRITICAL] Database setup failed: {init_err}.")