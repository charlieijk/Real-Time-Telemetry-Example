"""
3D Visualization for Solar Boat Racing.

This module provides:
1. 3D trajectory visualization with Plotly
2. Real-time 3D dashboard updates
3. Animated race replays
4. Terrain and water surface rendering
5. Multi-boat race visualization
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..ml.world_model import BoatState


@dataclass
class Boat3DConfig:
    """3D boat visualization configuration."""
    color: str = "#1f77b4"
    size: float = 10.0
    trail_length: int = 100
    show_velocity_vector: bool = True
    show_heading: bool = True
    opacity: float = 0.8


@dataclass
class Scene3DConfig:
    """3D scene configuration."""
    water_color: str = "#006994"
    sky_color: str = "#87CEEB"
    grid_color: str = "#444444"
    show_water_surface: bool = True
    show_grid: bool = True
    camera_distance: float = 500.0
    aspect_ratio: Tuple[float, float, float] = (1.0, 1.0, 0.3)


def create_3d_trajectory(
    trajectory: List[BoatState],
    config: Optional[Boat3DConfig] = None,
    show_velocity: bool = True,
    color_by: str = 'time'
) -> go.Figure:
    """
    Create 3D trajectory visualization.

    Args:
        trajectory: List of boat states
        config: Visualization configuration
        show_velocity: Show velocity vectors
        color_by: Color mapping ('time', 'velocity', 'soc', 'power')

    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required for 3D visualization")

    config = config or Boat3DConfig()

    # Extract trajectory data
    x = np.array([s.position[0] for s in trajectory])
    y = np.array([s.position[1] for s in trajectory])
    times = np.array([s.time for s in trajectory])
    velocities = np.array([s.velocity for s in trajectory])
    socs = np.array([s.battery_soc for s in trajectory])

    # Z-axis based on velocity or time
    z = velocities * 10  # Scale for visibility

    # Color mapping
    if color_by == 'velocity':
        colors = velocities
        colorbar_title = 'Velocity (m/s)'
    elif color_by == 'soc':
        colors = socs
        colorbar_title = 'Battery SOC'
    elif color_by == 'power':
        colors = np.array([s.solar_power for s in trajectory])
        colorbar_title = 'Solar Power (W)'
    else:  # time
        colors = times
        colorbar_title = 'Time (s)'

    # Create figure
    fig = go.Figure()

    # Main trajectory line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title=colorbar_title),
            opacity=config.opacity
        ),
        line=dict(
            color=colors,
            colorscale='Viridis',
            width=3
        ),
        name='Trajectory',
        hovertemplate=(
            'X: %{x:.1f}m<br>'
            'Y: %{y:.1f}m<br>'
            'Velocity: %{z:.2f}m/s<br>'
            '<extra></extra>'
        )
    ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='Start',
        hoverinfo='name'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='square'),
        name='End',
        hoverinfo='name'
    ))

    # Velocity vectors (every 10th point)
    if show_velocity and len(trajectory) > 10:
        step = max(1, len(trajectory) // 20)
        for i in range(0, len(trajectory), step):
            if i >= len(trajectory):
                continue

            state = trajectory[i]
            vx = state.velocity * np.cos(state.heading) * 5
            vy = state.velocity * np.sin(state.heading) * 5

            fig.add_trace(go.Cone(
                x=[x[i]], y=[y[i]], z=[z[i]],
                u=[vx], v=[vy], w=[0],
                colorscale=[[0, 'orange'], [1, 'orange']],
                showscale=False,
                sizemode='absolute',
                sizeref=3,
                opacity=0.6,
                hoverinfo='skip'
            ))

    # Water surface (at z=0)
    water_x = np.linspace(x.min() - 50, x.max() + 50, 20)
    water_y = np.linspace(y.min() - 50, y.max() + 50, 20)
    water_X, water_Y = np.meshgrid(water_x, water_y)
    water_Z = np.zeros_like(water_X)

    fig.add_trace(go.Surface(
        x=water_X, y=water_Y, z=water_Z,
        colorscale=[[0, 'rgba(0, 105, 148, 0.3)'], [1, 'rgba(0, 150, 200, 0.3)']],
        showscale=False,
        hoverinfo='skip',
        name='Water'
    ))

    # Update layout
    fig.update_layout(
        title='3D Race Trajectory',
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Velocity × 10 (m/s)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def create_3d_state_space(
    trajectory: List[BoatState],
    axes: Tuple[str, str, str] = ('velocity', 'soc', 'power')
) -> go.Figure:
    """
    Create 3D state-space visualization.

    Useful for understanding system dynamics and control strategies.

    Args:
        trajectory: List of boat states
        axes: Which state variables to plot (x, y, z)

    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required")

    # Extract data based on axes specification
    def get_axis_data(axis_name: str) -> Tuple[np.ndarray, str]:
        if axis_name == 'velocity':
            return np.array([s.velocity for s in trajectory]), 'Velocity (m/s)'
        elif axis_name == 'soc':
            return np.array([s.battery_soc for s in trajectory]), 'Battery SOC'
        elif axis_name == 'power':
            return np.array([s.solar_power for s in trajectory]), 'Solar Power (W)'
        elif axis_name == 'current':
            return np.array([s.motor_current for s in trajectory]), 'Motor Current (A)'
        elif axis_name == 'voltage':
            return np.array([s.battery_voltage for s in trajectory]), 'Voltage (V)'
        elif axis_name == 'time':
            return np.array([s.time for s in trajectory]), 'Time (s)'
        elif axis_name == 'position_x':
            return np.array([s.position[0] for s in trajectory]), 'X Position (m)'
        elif axis_name == 'position_y':
            return np.array([s.position[1] for s in trajectory]), 'Y Position (m)'
        else:
            return np.zeros(len(trajectory)), 'Unknown'

    x_data, x_label = get_axis_data(axes[0])
    y_data, y_label = get_axis_data(axes[1])
    z_data, z_label = get_axis_data(axes[2])

    times = np.array([s.time for s in trajectory])

    fig = go.Figure()

    # State trajectory
    fig.add_trace(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode='lines+markers',
        marker=dict(
            size=3,
            color=times,
            colorscale='Plasma',
            colorbar=dict(title='Time (s)'),
            opacity=0.8
        ),
        line=dict(
            color='rgba(100, 100, 100, 0.5)',
            width=2
        ),
        name='State Trajectory'
    ))

    # Start and end
    fig.add_trace(go.Scatter3d(
        x=[x_data[0]], y=[y_data[0]], z=[z_data[0]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='diamond'),
        name='Start'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_data[-1]], y=[y_data[-1]], z=[z_data[-1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='End'
    ))

    fig.update_layout(
        title=f'3D State Space: {axes[0]} vs {axes[1]} vs {axes[2]}',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            aspectmode='auto'
        ),
        showlegend=True
    )

    return fig


def create_animated_replay(
    trajectory: List[BoatState],
    fps: int = 30,
    trail_seconds: float = 10.0
) -> go.Figure:
    """
    Create animated 3D replay of race.

    Args:
        trajectory: List of boat states
        fps: Animation frames per second
        trail_seconds: Length of trailing path

    Returns:
        Plotly figure with animation
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required")

    x = np.array([s.position[0] for s in trajectory])
    y = np.array([s.position[1] for s in trajectory])
    velocities = np.array([s.velocity for s in trajectory])
    times = np.array([s.time for s in trajectory])

    z = velocities * 10

    # Determine frame step based on data length
    n_frames = min(len(trajectory), 200)
    frame_step = max(1, len(trajectory) // n_frames)

    frames = []
    for i in range(0, len(trajectory), frame_step):
        # Trail window
        dt = trajectory[i].time - trajectory[0].time if i > 0 else 0
        trail_start = max(0, i - int(trail_seconds * 10))

        frame_data = [
            # Current position
            go.Scatter3d(
                x=[x[i]], y=[y[i]], z=[z[i]],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Current'
            ),
            # Trail
            go.Scatter3d(
                x=x[trail_start:i+1],
                y=y[trail_start:i+1],
                z=z[trail_start:i+1],
                mode='lines',
                line=dict(color='blue', width=3),
                name='Trail'
            )
        ]

        frames.append(go.Frame(data=frame_data, name=f'frame_{i}'))

    # Create base figure
    fig = go.Figure(
        data=[
            # Water surface
            go.Surface(
                x=np.linspace(x.min()-50, x.max()+50, 10).reshape(1, -1).repeat(10, axis=0),
                y=np.linspace(y.min()-50, y.max()+50, 10).reshape(-1, 1).repeat(10, axis=1),
                z=np.zeros((10, 10)),
                colorscale=[[0, 'rgba(0, 105, 148, 0.3)'], [1, 'rgba(0, 150, 200, 0.3)']],
                showscale=False,
                hoverinfo='skip'
            ),
            # Initial boat position
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Boat'
            ),
            # Full path (faded)
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                name='Full Path'
            )
        ],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title='Race Replay',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Velocity × 10',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.1,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1000/fps, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )
                        ]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate'
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method='animate',
                        args=[
                            [f'frame_{i}'],
                            dict(mode='immediate', frame=dict(duration=0))
                        ],
                        label=f'{trajectory[i].time:.0f}s'
                    )
                    for i in range(0, len(trajectory), frame_step)
                ],
                x=0.1,
                len=0.8,
                y=0,
                currentvalue=dict(prefix='Time: ', suffix='s')
            )
        ]
    )

    return fig


def create_multi_boat_3d(
    trajectories: Dict[str, List[BoatState]],
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Create 3D visualization with multiple boats.

    Args:
        trajectories: Dictionary of boat_name -> trajectory
        colors: Optional color mapping

    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required")

    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]

    fig = go.Figure()

    for i, (boat_name, trajectory) in enumerate(trajectories.items()):
        color = (colors or {}).get(boat_name, default_colors[i % len(default_colors)])

        x = np.array([s.position[0] for s in trajectory])
        y = np.array([s.position[1] for s in trajectory])
        z = np.array([s.velocity for s in trajectory]) * 10

        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=3, color=color, opacity=0.7),
            line=dict(color=color, width=3),
            name=boat_name
        ))

        # Current position (end of trajectory)
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(size=12, color=color, symbol='diamond'),
            name=f'{boat_name} (current)',
            showlegend=False
        ))

    # Water surface
    all_x = np.concatenate([np.array([s.position[0] for s in t]) for t in trajectories.values()])
    all_y = np.concatenate([np.array([s.position[1] for s in t]) for t in trajectories.values()])

    water_x = np.linspace(all_x.min() - 50, all_x.max() + 50, 20)
    water_y = np.linspace(all_y.min() - 50, all_y.max() + 50, 20)
    water_X, water_Y = np.meshgrid(water_x, water_y)

    fig.add_trace(go.Surface(
        x=water_X, y=water_Y, z=np.zeros_like(water_X),
        colorscale=[[0, 'rgba(0, 105, 148, 0.3)'], [1, 'rgba(0, 150, 200, 0.3)']],
        showscale=False,
        hoverinfo='skip',
        name='Water'
    ))

    fig.update_layout(
        title='Multi-Boat Race Visualization',
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Velocity × 10 (m/s)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_telemetry_3d_dashboard(
    trajectory: List[BoatState]
) -> go.Figure:
    """
    Create comprehensive 3D dashboard with multiple views.

    Combines:
    - 3D trajectory
    - State-space plot
    - Time series
    - Performance metrics
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required")

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scene'}, {'type': 'scene'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ],
        subplot_titles=[
            '3D Trajectory',
            'State Space (V, SOC, Power)',
            'Velocity Profile',
            'Battery State'
        ]
    )

    # Extract data
    x = np.array([s.position[0] for s in trajectory])
    y = np.array([s.position[1] for s in trajectory])
    times = np.array([s.time for s in trajectory])
    velocities = np.array([s.velocity for s in trajectory])
    socs = np.array([s.battery_soc for s in trajectory])
    powers = np.array([s.solar_power for s in trajectory])
    currents = np.array([s.motor_current for s in trajectory])

    # 1. 3D Trajectory (row 1, col 1)
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=velocities * 10,
            mode='lines',
            line=dict(color=times, colorscale='Viridis', width=4),
            name='Trajectory'
        ),
        row=1, col=1
    )

    # 2. State Space (row 1, col 2)
    fig.add_trace(
        go.Scatter3d(
            x=velocities, y=socs, z=powers,
            mode='lines+markers',
            marker=dict(size=3, color=times, colorscale='Plasma'),
            line=dict(color='gray', width=1),
            name='State Space'
        ),
        row=1, col=2
    )

    # 3. Velocity Profile (row 2, col 1)
    fig.add_trace(
        go.Scatter(x=times, y=velocities, mode='lines', name='Velocity', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=currents, mode='lines', name='Current', line=dict(color='orange'), yaxis='y3'),
        row=2, col=1
    )

    # 4. Battery State (row 2, col 2)
    fig.add_trace(
        go.Scatter(x=times, y=socs * 100, mode='lines', name='SOC %', line=dict(color='green')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=times, y=powers, mode='lines', name='Solar Power', line=dict(color='yellow'), yaxis='y4'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text='Solar Regatta 3D Telemetry Dashboard'
    )

    # Update scene layouts
    fig.update_scenes(
        dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='V × 10'
        ),
        row=1, col=1
    )

    fig.update_scenes(
        dict(
            xaxis_title='Velocity',
            yaxis_title='SOC',
            zaxis_title='Power'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_xaxes(title_text='Time (s)', row=2, col=2)
    fig.update_yaxes(title_text='Velocity (m/s)', row=2, col=1)
    fig.update_yaxes(title_text='SOC (%)', row=2, col=2)

    return fig


def export_3d_to_html(fig: go.Figure, output_path: str):
    """Export 3D figure to standalone HTML file."""
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def export_3d_to_json(fig: go.Figure, output_path: str):
    """Export 3D figure data to JSON for web embedding."""
    with open(output_path, 'w') as f:
        f.write(fig.to_json())
