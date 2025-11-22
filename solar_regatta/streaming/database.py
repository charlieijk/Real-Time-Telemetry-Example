"""
Database Backend for Solar Regatta Telemetry.

Provides:
1. SQLite backend for local storage
2. Time-series optimized schema
3. Session management for races
4. Query interface for analysis
5. Export capabilities (CSV, JSON, Parquet)
"""
from __future__ import annotations

import sqlite3
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import threading

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = "telemetry.db"
    enable_wal: bool = True  # Write-ahead logging for performance
    batch_size: int = 100    # Batch insert size
    auto_vacuum: bool = True


@dataclass
class RaceSession:
    """Race session metadata."""
    session_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    boat_name: str = "default"
    location: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    n_samples: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TelemetryRecord:
    """Single telemetry record for database storage."""
    timestamp: float
    session_id: str

    # Motor data
    motor_current: float = 0.0
    motor_rpm: float = 0.0
    duty_cycle: float = 0.0
    motor_temp: float = 0.0

    # Battery data
    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_soc: float = 0.0
    battery_temp: float = 0.0

    # GPS data
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed_gps: float = 0.0
    heading: float = 0.0

    # Computed
    speed_computed: float = 0.0
    distance: float = 0.0
    power: float = 0.0

    # Environment
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    sun_intensity: float = 0.0

    # Extra JSON data
    extra: str = "{}"


class TelemetryDatabase:
    """
    SQLite-based telemetry database with time-series optimizations.

    Features:
    - WAL mode for concurrent read/write
    - Batch inserts for performance
    - Indexed queries by time and session
    - Aggregation functions for analysis
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        name TEXT,
        start_time REAL,
        end_time REAL,
        boat_name TEXT,
        location TEXT,
        conditions TEXT,
        notes TEXT,
        n_samples INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        session_id TEXT NOT NULL,
        motor_current REAL,
        motor_rpm REAL,
        duty_cycle REAL,
        motor_temp REAL,
        battery_voltage REAL,
        battery_current REAL,
        battery_soc REAL,
        battery_temp REAL,
        latitude REAL,
        longitude REAL,
        altitude REAL,
        speed_gps REAL,
        heading REAL,
        speed_computed REAL,
        distance REAL,
        power REAL,
        wind_speed REAL,
        wind_direction REAL,
        sun_intensity REAL,
        extra TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry(timestamp);
    CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry(session_id);
    CREATE INDEX IF NOT EXISTS idx_telemetry_session_time ON telemetry(session_id, timestamp);

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        session_id TEXT NOT NULL,
        model_name TEXT,
        predicted_speed REAL,
        predicted_soc REAL,
        predicted_distance REAL,
        confidence REAL,
        extra TEXT
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        session_id TEXT,
        event_type TEXT,
        severity TEXT,
        message TEXT,
        data TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._local = threading.local()
        self._write_buffer: List[TelemetryRecord] = []
        self._buffer_lock = threading.Lock()

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.config.db_path,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row

            if self.config.enable_wal:
                self._local.conn.execute("PRAGMA journal_mode=WAL")

            if self.config.auto_vacuum:
                self._local.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Session Management

    def create_session(
        self,
        name: str,
        boat_name: str = "default",
        location: str = "",
        conditions: Optional[Dict] = None,
        notes: str = ""
    ) -> RaceSession:
        """Create new race session."""
        session_id = f"session_{int(time.time() * 1000)}"
        start_time = time.time()

        session = RaceSession(
            session_id=session_id,
            name=name,
            start_time=start_time,
            boat_name=boat_name,
            location=location,
            conditions=conditions or {},
            notes=notes
        )

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                (session_id, name, start_time, boat_name, location, conditions, notes, n_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    session.session_id,
                    session.name,
                    session.start_time,
                    session.boat_name,
                    session.location,
                    json.dumps(session.conditions),
                    session.notes
                )
            )

        return session

    def end_session(self, session_id: str):
        """End a race session."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE sessions SET end_time = ? WHERE session_id = ?",
                (time.time(), session_id)
            )

    def get_session(self, session_id: str) -> Optional[RaceSession]:
        """Get session by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if row:
            return RaceSession(
                session_id=row['session_id'],
                name=row['name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                boat_name=row['boat_name'],
                location=row['location'],
                conditions=json.loads(row['conditions'] or '{}'),
                notes=row['notes'],
                n_samples=row['n_samples']
            )
        return None

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[RaceSession]:
        """List all sessions."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM sessions
            ORDER BY start_time DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()

        return [
            RaceSession(
                session_id=row['session_id'],
                name=row['name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                boat_name=row['boat_name'],
                location=row['location'],
                conditions=json.loads(row['conditions'] or '{}'),
                notes=row['notes'],
                n_samples=row['n_samples']
            )
            for row in rows
        ]

    # Telemetry Data

    def insert_telemetry(self, record: TelemetryRecord):
        """Insert single telemetry record (buffered)."""
        with self._buffer_lock:
            self._write_buffer.append(record)

            if len(self._write_buffer) >= self.config.batch_size:
                self._flush_buffer()

    def insert_telemetry_batch(self, records: List[TelemetryRecord]):
        """Insert batch of telemetry records."""
        if not records:
            return

        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO telemetry
                (timestamp, session_id, motor_current, motor_rpm, duty_cycle, motor_temp,
                 battery_voltage, battery_current, battery_soc, battery_temp,
                 latitude, longitude, altitude, speed_gps, heading,
                 speed_computed, distance, power, wind_speed, wind_direction,
                 sun_intensity, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.timestamp, r.session_id, r.motor_current, r.motor_rpm,
                        r.duty_cycle, r.motor_temp, r.battery_voltage, r.battery_current,
                        r.battery_soc, r.battery_temp, r.latitude, r.longitude,
                        r.altitude, r.speed_gps, r.heading, r.speed_computed,
                        r.distance, r.power, r.wind_speed, r.wind_direction,
                        r.sun_intensity, r.extra
                    )
                    for r in records
                ]
            )

            # Update session sample counts
            session_counts: Dict[str, int] = {}
            for r in records:
                session_counts[r.session_id] = session_counts.get(r.session_id, 0) + 1

            for session_id, count in session_counts.items():
                conn.execute(
                    "UPDATE sessions SET n_samples = n_samples + ? WHERE session_id = ?",
                    (count, session_id)
                )

    def _flush_buffer(self):
        """Flush write buffer to database."""
        if not self._write_buffer:
            return

        records = self._write_buffer.copy()
        self._write_buffer.clear()

        self.insert_telemetry_batch(records)

    def flush(self):
        """Force flush write buffer."""
        with self._buffer_lock:
            self._flush_buffer()

    def query_telemetry(
        self,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        columns: Optional[List[str]] = None,
        limit: int = 10000,
        downsample_factor: int = 1
    ) -> List[Dict]:
        """
        Query telemetry data.

        Args:
            session_id: Filter by session
            start_time: Start timestamp
            end_time: End timestamp
            columns: Columns to return (None = all)
            limit: Maximum records
            downsample_factor: Return every Nth record

        Returns:
            List of telemetry records as dictionaries
        """
        self.flush()  # Ensure buffer is written

        conn = self._get_connection()

        col_str = "*" if not columns else ", ".join(columns)

        query = f"SELECT {col_str} FROM telemetry WHERE 1=1"
        params: List[Any] = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if downsample_factor > 1:
            query += f" AND id % {downsample_factor} = 0"

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated statistics for a session."""
        self.flush()

        conn = self._get_connection()

        stats = conn.execute(
            """
            SELECT
                COUNT(*) as n_samples,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                AVG(speed_computed) as avg_speed,
                MAX(speed_computed) as max_speed,
                AVG(battery_voltage) as avg_voltage,
                MIN(battery_soc) as min_soc,
                MAX(battery_soc) as max_soc,
                AVG(motor_current) as avg_current,
                MAX(motor_current) as max_current,
                MAX(distance) as total_distance,
                AVG(power) as avg_power
            FROM telemetry
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

        return dict(stats) if stats else {}

    # Export Functions

    def export_to_csv(
        self,
        session_id: str,
        output_path: str,
        columns: Optional[List[str]] = None
    ):
        """Export session telemetry to CSV."""
        if not HAS_PANDAS:
            # Fallback without pandas
            data = self.query_telemetry(session_id=session_id, limit=1000000)
            if not data:
                return

            import csv
            cols = columns or list(data[0].keys())

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k) for k in cols})
        else:
            data = self.query_telemetry(session_id=session_id, limit=1000000)
            df = pd.DataFrame(data)
            if columns:
                df = df[columns]
            df.to_csv(output_path, index=False)

    def export_to_json(
        self,
        session_id: str,
        output_path: str
    ):
        """Export session telemetry to JSON."""
        session = self.get_session(session_id)
        data = self.query_telemetry(session_id=session_id, limit=1000000)
        stats = self.get_session_stats(session_id)

        export = {
            'session': session.to_dict() if session else None,
            'stats': stats,
            'telemetry': data
        }

        with open(output_path, 'w') as f:
            json.dump(export, f, indent=2)

    def to_dataframe(
        self,
        session_id: str,
        columns: Optional[List[str]] = None
    ):
        """Convert session telemetry to pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")

        data = self.query_telemetry(session_id=session_id, limit=1000000)
        df = pd.DataFrame(data)

        if columns:
            df = df[columns]

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        return df

    # Events/Anomalies

    def log_event(
        self,
        event_type: str,
        message: str,
        session_id: Optional[str] = None,
        severity: str = "info",
        data: Optional[Dict] = None
    ):
        """Log event or anomaly."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO events (timestamp, session_id, event_type, severity, message, data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    session_id,
                    event_type,
                    severity,
                    message,
                    json.dumps(data or {})
                )
            )

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query events."""
        conn = self._get_connection()

        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # Cleanup

    def vacuum(self):
        """Optimize database storage."""
        conn = self._get_connection()
        conn.execute("PRAGMA incremental_vacuum")
        conn.execute("VACUUM")

    def delete_session(self, session_id: str):
        """Delete session and all its data."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM telemetry WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM predictions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def close(self):
        """Close database connection."""
        self.flush()
        if hasattr(self._local, 'conn'):
            self._local.conn.close()


def create_database(
    db_path: str = "telemetry.db",
    **kwargs
) -> TelemetryDatabase:
    """Create database with specified configuration."""
    config = DatabaseConfig(db_path=db_path, **kwargs)
    return TelemetryDatabase(config)
