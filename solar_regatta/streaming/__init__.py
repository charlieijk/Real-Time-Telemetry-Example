"""
Real-time Streaming and Data Infrastructure for Solar Regatta.

Provides:
- WebSocket server for live telemetry streaming
- Database backend for time-series storage
- Event-driven architecture for real-time updates
"""

from .realtime import (
    TelemetryEvent,
    StreamingServer,
    TelemetryPublisher,
    TelemetrySubscriber,
    create_streaming_server,
)

from .database import (
    DatabaseConfig,
    TelemetryDatabase,
    RaceSession,
    create_database,
)

__all__ = [
    'TelemetryEvent',
    'StreamingServer',
    'TelemetryPublisher',
    'TelemetrySubscriber',
    'create_streaming_server',
    'DatabaseConfig',
    'TelemetryDatabase',
    'RaceSession',
    'create_database',
]
