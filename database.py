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


def init_db():
    """Initializes the SQLite database and creates all persistent tables."""
    # Ensure parent directories exist (crucial for local/fallback path variations)
    parent_dir = os.path.dirname(DB_PATH)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Use timeout parameters to gracefully manage network latency on HF Storage Buckets
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()

    # Enable WAL mode for optimized concurrent reading/writing over FUSE storage layers
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        pass # Fallback cleanly if the FUSE layer restricts WAL allocation

    # -----------------------------------------------------------------------
    # call_logs — one row per WebSocket session (inbound phone call)
    # -----------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT    UNIQUE NOT NULL,
            provider         TEXT    NOT NULL,
            language         TEXT    NOT NULL,
            caller_number    TEXT    DEFAULT 'UNKNOWN',
            start_time       TIMESTAMP NOT NULL,
            end_time         TIMESTAMP,
            duration_seconds REAL,
            status           TEXT    DEFAULT 'active'
        )
    """)
    # Fast time-series queries for the dashboard frequency/duration charts
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time   ON call_logs(start_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_caller       ON call_logs(caller_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider     ON call_logs(provider)")

    # -----------------------------------------------------------------------
    # model_load_logs — records how long each model took to load at startup
    # -----------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_load_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name   TEXT    NOT NULL,
            duration_s   REAL    NOT NULL,
            device       TEXT    DEFAULT 'cpu',
            loaded_at    TIMESTAMP NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loaded_at ON model_load_logs(loaded_at)")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Call record helpers
# ---------------------------------------------------------------------------

def db_start_call(session_id: str, provider: str, language: str, caller_number: str = "UNKNOWN"):
    """Inserts a new call record when a WebSocket stream initialises."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO call_logs
                (session_id, provider, language, caller_number, start_time, status)
            VALUES (?, ?, ?, ?, ?, 'active')
        """, (session_id, provider.lower(), language.lower(),
              caller_number, datetime.utcnow()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Guard against duplicate WebSocket frames
    except Exception as e:
        print(f"[database] Failed to record call start: {e}")
    finally:
        conn.close()


def db_end_call(session_id: str, completed: bool = True):
    """Stamps end_time, duration, and final status onto an existing call row."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    now = datetime.utcnow()

    try:
        cursor.execute("SELECT start_time FROM call_logs WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if row:
            raw = row[0]
            if isinstance(raw, str):
                try:
                    start_time = datetime.fromisoformat(raw)
                except ValueError:
                    start_time = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
            else:
                start_time = raw

            duration = (now - start_time).total_seconds()
            status   = "completed" if completed else "failed"

            cursor.execute("""
                UPDATE call_logs
                   SET end_time         = ?,
                       duration_seconds = ?,
                       status           = ?
                 WHERE session_id = ?
            """, (now, round(duration, 2), status, session_id))
            conn.commit()
    except Exception as e:
        print(f"[database] Failed to log call end execution: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model load helper
# ---------------------------------------------------------------------------

def db_log_model_load(model_name: str, duration_s: float, device: str = "cpu"):
    """Persists how long a specific model took to initialise at startup."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO model_load_logs (model_name, duration_s, device, loaded_at)
            VALUES (?, ?, ?, ?)
        """, (model_name, round(duration_s, 3), device.lower(), datetime.utcnow()))
        conn.commit()
    except Exception as e:
        print(f"[database] Failed to log model load: {e}")
    finally:
        conn.close()



# Async wrappers — keeps FastAPI's event loop unblocked during disk I/O
# ---------------------------------------------------------------------------

async def async_start_call(session_id: str, provider: str, language: str,
                           caller_number: str = "UNKNOWN"):
    await asyncio.to_thread(db_start_call, session_id, provider, language, caller_number)


async def async_end_call(session_id: str, completed: bool = True):
    await asyncio.to_thread(db_end_call, session_id, completed)


async def async_log_model_load(model_name: str, duration_s: float, device: str = "cpu"):
    await asyncio.to_thread(db_log_model_load, model_name, duration_s, device)


# Safely execute tables verification block
try:
    init_db()
except Exception as init_err:
    print(f"[CRITICAL] Database setup failed: {init_err}. Retrying sequentially on active calls.")