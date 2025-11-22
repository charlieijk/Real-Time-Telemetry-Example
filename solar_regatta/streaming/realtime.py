"""
Real-time Telemetry Streaming Infrastructure.

This module provides:
1. WebSocket server for live data streaming
2. Pub/Sub event system for real-time updates
3. Data buffering and windowing
4. Client connection management
5. Integration with VESC hardware collectors
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Set
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Queue, Empty
import weakref

try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

import numpy as np


@dataclass
class TelemetryEvent:
    """Single telemetry event."""
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    source: str = "vesc"
    sequence: int = 0

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'TelemetryEvent':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class TelemetryFrame:
    """
    Aggregated telemetry frame containing multiple metrics.

    Sent at regular intervals (e.g., 10Hz).
    """
    timestamp: float
    frame_id: int

    # Motor/VESC data
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
    speed_gps: float = 0.0
    heading: float = 0.0

    # Computed metrics
    speed_computed: float = 0.0
    distance: float = 0.0
    power: float = 0.0
    efficiency: float = 0.0

    # Predictions
    predicted_speed: Optional[float] = None
    predicted_soc: Optional[float] = None
    mpc_control: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class EventBus:
    """
    Simple pub/sub event bus for internal communication.

    Thread-safe event distribution to multiple subscribers.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type]
                    if cb != callback
                ]

    def publish(self, event: TelemetryEvent):
        """Publish event to all subscribers."""
        with self._lock:
            callbacks = self._subscribers.get(event.event_type, []).copy()
            callbacks.extend(self._subscribers.get('*', []))  # Wildcard subscribers

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")


class TelemetryBuffer:
    """
    Ring buffer for telemetry data with windowing support.

    Provides efficient storage and retrieval of recent telemetry.
    """

    def __init__(self, max_size: int = 10000, window_seconds: float = 60.0):
        self.max_size = max_size
        self.window_seconds = window_seconds
        self._buffer: List[TelemetryFrame] = []
        self._lock = threading.Lock()

    def append(self, frame: TelemetryFrame):
        """Add frame to buffer."""
        with self._lock:
            self._buffer.append(frame)
            if len(self._buffer) > self.max_size:
                self._buffer.pop(0)

    def get_window(self, seconds: Optional[float] = None) -> List[TelemetryFrame]:
        """Get frames from last N seconds."""
        seconds = seconds or self.window_seconds
        cutoff = time.time() - seconds

        with self._lock:
            return [f for f in self._buffer if f.timestamp >= cutoff]

    def get_latest(self, n: int = 1) -> List[TelemetryFrame]:
        """Get last N frames."""
        with self._lock:
            return self._buffer[-n:]

    def get_all(self) -> List[TelemetryFrame]:
        """Get all buffered frames."""
        with self._lock:
            return self._buffer.copy()

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class TelemetryPublisher:
    """
    Publishes telemetry data to connected clients.

    Handles:
    - Frame aggregation and rate limiting
    - Client connection management
    - Broadcast to all connected clients
    """

    def __init__(self, target_fps: float = 10.0):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.event_bus = EventBus()
        self.buffer = TelemetryBuffer()

        self._frame_id = 0
        self._last_frame_time = 0.0
        self._current_frame = TelemetryFrame(timestamp=0.0, frame_id=0)
        self._lock = threading.Lock()

        # Websocket clients
        self._clients: Set[Any] = set()

    def update_telemetry(
        self,
        motor_current: Optional[float] = None,
        motor_rpm: Optional[float] = None,
        battery_voltage: Optional[float] = None,
        battery_soc: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        speed: Optional[float] = None,
        **kwargs
    ):
        """Update current telemetry values."""
        with self._lock:
            if motor_current is not None:
                self._current_frame.motor_current = motor_current
            if motor_rpm is not None:
                self._current_frame.motor_rpm = motor_rpm
            if battery_voltage is not None:
                self._current_frame.battery_voltage = battery_voltage
            if battery_soc is not None:
                self._current_frame.battery_soc = battery_soc
            if latitude is not None:
                self._current_frame.latitude = latitude
            if longitude is not None:
                self._current_frame.longitude = longitude
            if speed is not None:
                self._current_frame.speed_computed = speed

            # Handle additional fields
            for key, value in kwargs.items():
                if hasattr(self._current_frame, key):
                    setattr(self._current_frame, key, value)

    def tick(self) -> Optional[TelemetryFrame]:
        """
        Check if it's time to publish a new frame.

        Returns frame if published, None otherwise.
        """
        current_time = time.time()

        if current_time - self._last_frame_time >= self.frame_interval:
            with self._lock:
                self._frame_id += 1
                self._current_frame.timestamp = current_time
                self._current_frame.frame_id = self._frame_id

                # Create copy for publishing
                frame = TelemetryFrame(**asdict(self._current_frame))

            # Store in buffer
            self.buffer.append(frame)

            # Publish event
            event = TelemetryEvent(
                timestamp=current_time,
                event_type='telemetry_frame',
                data=frame.to_dict(),
                sequence=self._frame_id
            )
            self.event_bus.publish(event)

            self._last_frame_time = current_time
            return frame

        return None

    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSocket clients."""
        if not self._clients:
            return

        # Create tasks for all clients
        tasks = []
        for client in self._clients.copy():
            try:
                tasks.append(asyncio.create_task(client.send(message)))
            except Exception:
                self._clients.discard(client)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def register_client(self, websocket):
        """Register new WebSocket client."""
        self._clients.add(websocket)

    def unregister_client(self, websocket):
        """Unregister WebSocket client."""
        self._clients.discard(websocket)


class TelemetrySubscriber:
    """
    Subscribes to telemetry stream and processes data.

    Use for:
    - Real-time visualization
    - Data logging
    - Alert monitoring
    """

    def __init__(self, publisher: TelemetryPublisher):
        self.publisher = publisher
        self._callbacks: List[Callable[[TelemetryFrame], None]] = []

        # Subscribe to publisher events
        publisher.event_bus.subscribe('telemetry_frame', self._on_frame)

    def _on_frame(self, event: TelemetryEvent):
        """Handle incoming telemetry frame."""
        frame = TelemetryFrame(**event.data)
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                print(f"Subscriber callback error: {e}")

    def on_telemetry(self, callback: Callable[[TelemetryFrame], None]):
        """Register callback for telemetry frames."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove callback."""
        self._callbacks = [cb for cb in self._callbacks if cb != callback]


class StreamingServer:
    """
    WebSocket server for real-time telemetry streaming.

    Provides:
    - WebSocket endpoint for clients
    - JSON protocol for telemetry data
    - Command handling from clients
    - Connection management
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        publisher: Optional[TelemetryPublisher] = None
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.host = host
        self.port = port
        self.publisher = publisher or TelemetryPublisher()

        self._running = False
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _handler(self, websocket, path):
        """Handle WebSocket connection."""
        self.publisher.register_client(websocket)
        client_addr = websocket.remote_address

        try:
            # Send initial state
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to Solar Regatta Telemetry',
                'timestamp': time.time()
            }))

            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.publisher.unregister_client(websocket)

    async def _handle_message(self, websocket, message: str):
        """Handle incoming client message."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')

            if msg_type == 'subscribe':
                # Client wants to subscribe to specific events
                await websocket.send(json.dumps({
                    'type': 'subscribed',
                    'topics': data.get('topics', ['telemetry'])
                }))

            elif msg_type == 'command':
                # Handle control commands
                await self._handle_command(websocket, data)

            elif msg_type == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': time.time()
                }))

            elif msg_type == 'get_history':
                # Send historical data
                window = data.get('seconds', 60)
                frames = self.publisher.buffer.get_window(window)
                await websocket.send(json.dumps({
                    'type': 'history',
                    'frames': [f.to_dict() for f in frames]
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON'
            }))

    async def _handle_command(self, websocket, data: Dict):
        """Handle control command from client."""
        command = data.get('command', '')

        if command == 'set_motor_current':
            # Forward to hardware controller
            target_current = data.get('value', 0.0)
            # In production: self.hardware_controller.set_current(target_current)
            await websocket.send(json.dumps({
                'type': 'command_ack',
                'command': command,
                'value': target_current
            }))

        elif command == 'start_recording':
            await websocket.send(json.dumps({
                'type': 'command_ack',
                'command': command,
                'status': 'recording_started'
            }))

        elif command == 'stop_recording':
            await websocket.send(json.dumps({
                'type': 'command_ack',
                'command': command,
                'status': 'recording_stopped'
            }))

    async def _broadcast_loop(self):
        """Periodically broadcast telemetry to clients."""
        while self._running:
            frame = self.publisher.tick()
            if frame:
                await self.publisher.broadcast(json.dumps({
                    'type': 'telemetry',
                    'frame': frame.to_dict()
                }))
            await asyncio.sleep(0.01)  # 100Hz check rate

    async def start_async(self):
        """Start server (async version)."""
        self._running = True

        # Start WebSocket server
        self._server = await serve(
            self._handler,
            self.host,
            self.port
        )

        # Start broadcast loop
        broadcast_task = asyncio.create_task(self._broadcast_loop())

        print(f"Telemetry server started on ws://{self.host}:{self.port}")

        try:
            await self._server.wait_closed()
        finally:
            self._running = False
            broadcast_task.cancel()

    def start(self):
        """Start server (blocking)."""
        asyncio.run(self.start_async())

    def start_background(self):
        """Start server in background thread."""
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.start_async())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()


class SimulatedTelemetrySource:
    """
    Simulated telemetry source for testing without hardware.

    Generates realistic telemetry data based on physics model.
    """

    def __init__(
        self,
        publisher: TelemetryPublisher,
        update_rate: float = 10.0
    ):
        self.publisher = publisher
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Simulated state
        self._velocity = 0.0
        self._position = np.array([0.0, 0.0])
        self._battery_soc = 1.0
        self._battery_voltage = 13.0
        self._motor_current = 0.0
        self._distance = 0.0
        self._time = 0.0

    def set_motor_current(self, current: float):
        """Set simulated motor current."""
        self._motor_current = np.clip(current, 0, 15)

    def _simulate_step(self, dt: float):
        """Simulate one time step."""
        # Simple physics
        motor_power = self._motor_current * self._battery_voltage
        thrust = motor_power * 0.5 / max(0.1, self._velocity)  # Simplified
        drag = 0.5 * 1000 * 0.5 * 0.3 * self._velocity ** 2
        net_force = thrust - drag
        acceleration = net_force / 50.0  # 50kg boat

        self._velocity = max(0, self._velocity + acceleration * dt)

        # Update position
        displacement = self._velocity * dt
        self._position[0] += displacement
        self._distance += displacement

        # Update battery
        power_used = motor_power * dt / 3600.0  # Wh
        self._battery_soc -= power_used / 100.0  # 100Wh battery
        self._battery_soc = max(0, self._battery_soc)
        self._battery_voltage = 13.0 * (0.8 + 0.2 * self._battery_soc)

        self._time += dt

        # Add some noise for realism
        noise = np.random.normal(0, 0.02)

    def _run(self):
        """Main simulation loop."""
        last_time = time.time()

        while self._running:
            current_time = time.time()
            dt = current_time - last_time

            if dt >= self.update_interval:
                self._simulate_step(dt)

                # Update publisher
                self.publisher.update_telemetry(
                    motor_current=self._motor_current,
                    battery_voltage=self._battery_voltage,
                    battery_soc=self._battery_soc,
                    speed=self._velocity,
                    latitude=40.0 + self._position[0] / 111000,
                    longitude=-74.0 + self._position[1] / 85000,
                    distance=self._distance
                )

                last_time = current_time

            time.sleep(0.001)

    def start(self):
        """Start simulation."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


def create_streaming_server(
    host: str = "localhost",
    port: int = 8765,
    simulated: bool = True
) -> StreamingServer:
    """
    Create and configure streaming server.

    Args:
        host: Server host
        port: Server port
        simulated: Use simulated telemetry source

    Returns:
        Configured streaming server
    """
    publisher = TelemetryPublisher(target_fps=10.0)
    server = StreamingServer(host, port, publisher)

    if simulated:
        source = SimulatedTelemetrySource(publisher)
        source.start()

    return server
