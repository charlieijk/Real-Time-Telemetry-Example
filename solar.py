import pyproj
import mgrs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from solar_regatta.ml import evaluate_model, prepare_training_data, train_speed_model

_GEOD = pyproj.Geod(ellps='WGS84')
_MGRS = mgrs.MGRS()


def dist(sp, ep):
    """Calculate distance and azimuth between two MGRS points"""
    try:
        # convert start point and end point
        (sLat, sLon) = _MGRS.toLatLon(sp)
        (eLat, eLon) = _MGRS.toLatLon(ep)

        # inv returns azimuth, back azimuth and distance
        a, a2, d = _GEOD.inv(sLon, sLat, eLon, eLat)
    except Exception as e:
        raise ValueError(f"Invalid MGRS point: {e}")
    else:
        return d, a


def calculate_speeds(gps_points, timestamps):
    """
    Calculate speed between consecutive GPS points

    Args:
        gps_points: list of MGRS coordinates or (lat, lon) tuples
        timestamps: list of datetime objects or seconds

    Returns:
        list of speeds in m/s
    """
    speeds = []

    for i in range(len(gps_points) - 1):
        try:
            distance, _ = dist(gps_points[i], gps_points[i + 1])

            # Calculate time difference in seconds
            if isinstance(timestamps[i], datetime):
                time_diff = (timestamps[i + 1] - timestamps[i]).total_seconds()
            else:
                time_diff = timestamps[i + 1] - timestamps[i]

            # Calculate speed (m/s)
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
            else:
                speeds.append(0)
        except Exception:
            speeds.append(0)

    return speeds


def plot_speed_vs_time(speeds, timestamps, title="Solar Boat Speed vs Time"):
    """
    Plot speed vs time

    Args:
        speeds: list of speeds in m/s
        timestamps: list of datetime objects or time values
        title: plot title
    """
    plt.figure(figsize=(12, 6))

    # Convert timestamps for plotting if needed
    if timestamps and isinstance(timestamps[0], datetime):
        time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
        xlabel = "Time (seconds)"
    else:
        time_values = timestamps
        xlabel = "Time"

    plt.plot(time_values[:-1], speeds, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel(xlabel)
    plt.ylabel("Speed (m/s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


def plot_with_coordinates(speeds, timestamps, gps_points, title="Speed vs Time with GPS Data"):
    """
    Plot speed vs time with GPS coordinates shown on hover (if using interactive backend)

    Args:
        speeds: list of speeds
        timestamps: list of timestamps
        gps_points: list of MGRS coordinates
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert timestamps for plotting
    if timestamps and isinstance(timestamps[0], datetime):
        time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
        xlabel = "Time (seconds)"
    else:
        time_values = timestamps
        xlabel = "Time"

    ax.plot(time_values[:-1], speeds, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add GPS coordinates to plot
    ax.text(0.02, 0.98, f"Start: {gps_points[0]}\nEnd: {gps_points[-1]}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    return fig, ax


# ============================================================================
# EXAMPLE: Generate Sample Data and Visualize
# ============================================================================

def generate_sample_vesc_data(duration_seconds=300, interval=5):
    """
    Generate realistic sample VESC telemetry data

    Args:
        duration_seconds: Total race duration in seconds (default 5 minutes)
        interval: Time between GPS samples in seconds (default 5 seconds)

    Returns:
        tuple: (gps_points, timestamps, speeds_raw, battery_voltage, motor_current)
    """
    import random

    gps_points = []
    timestamps = []
    speeds_raw = []
    battery_voltage = []
    motor_current = []

    start_time = datetime(2025, 10, 30, 14, 30, 0)

    # Simulate VESC data over time
    for i in range(0, duration_seconds + 1, interval):
        timestamp = start_time + timedelta(seconds=i)
        timestamps.append(timestamp)

        # Simulate GPS movement (slight variation in MGRS coordinates)
        offset = int(i * 10)  # Simulate moving north/east
        mgrs_coord = f"10SEG{6400000 + offset:010d}"
        gps_points.append(mgrs_coord)

        # Simulate speed variations (m/s)
        time_fraction = i / duration_seconds if duration_seconds > 0 else 0
        base_speed = 2.0 * (time_fraction if time_fraction < 0.5 else 1 - time_fraction)
        variation = random.gauss(0, 0.2)  # Add some noise
        speed = max(0, base_speed + variation)
        speeds_raw.append(speed)

        # Simulate battery voltage (12V system)
        voltage = 13.0 - (time_fraction * 1.5) + random.gauss(0, 0.1)
        battery_voltage.append(max(11.0, voltage))

        # Simulate motor current (amps)
        current = 5.0 * speed + random.gauss(0, 0.5)
        motor_current.append(max(0, current))

    return gps_points, timestamps, speeds_raw, battery_voltage, motor_current


def analyze_performance(speeds, battery_voltage, motor_current, timestamps):
    """
    Calculate performance statistics from VESC data

    Args:
        speeds: List of speed values in m/s
        battery_voltage: List of voltage readings
        motor_current: List of current readings
        timestamps: List of timestamps

    Returns:
        dict: Dictionary of performance metrics
    """
    if not speeds:
        return {}

    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    distance = sum(speeds) * (duration / len(speeds)) if len(speeds) > 0 else 0

    metrics = {
        'max_speed': max(speeds),
        'min_speed': min(speeds),
        'avg_speed': sum(speeds) / len(speeds),
        'distance': distance,
        'duration': duration,
        'max_voltage': max(battery_voltage),
        'min_voltage': min(battery_voltage),
        'max_current': max(motor_current),
        'avg_current': sum(motor_current) / len(motor_current),
    }

    return metrics


def plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points):
    """
    Create a comprehensive dashboard with multiple plots

    Args:
        speeds: List of speeds
        battery_voltage: List of voltages
        motor_current: List of currents
        timestamps: List of timestamps
        gps_points: List of GPS coordinates
    """
    # Convert timestamps to seconds from start
    time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Solar Boat Race Telemetry Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Speed vs Time
    ax1.plot(time_values[:-1], speeds, marker='o', linestyle='-', linewidth=2,
             markersize=5, color='#2E86AB')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Speed vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(time_values[:-1], speeds, alpha=0.2, color='#2E86AB')

    # Plot 2: Battery Voltage
    ax2.plot(time_values, battery_voltage, marker='s', linestyle='-', linewidth=2,
             markersize=5, color='#A23B72')
    ax2.axhline(y=11.0, color='r', linestyle='--', label='Low Voltage Cutoff')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Battery Voltage Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Motor Current
    ax3.plot(time_values, motor_current, marker='^', linestyle='-', linewidth=2,
             markersize=5, color='#F18F01')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Current (A)')
    ax3.set_title('Motor Current Draw')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(time_values, motor_current, alpha=0.2, color='#F18F01')

    # Plot 4: Speed vs Current (efficiency)
    ax4.scatter(speeds, motor_current[:-1], alpha=0.6, s=100, color='#C73E1D')
    ax4.set_xlabel('Speed (m/s)')
    ax4.set_ylabel('Current (A)')
    ax4.set_title('Speed vs Motor Current (Efficiency)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN: Run Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLAR BOAT RACE ANALYSIS - COMPLETE EXAMPLE")
    print("=" * 70)

    # Generate sample data
    print("\n[1] Generating sample VESC telemetry data (5 minute race)...")
    gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
        generate_sample_vesc_data(duration_seconds=300, interval=5)
    print(f"    Generated {len(gps_points)} GPS points with telemetry")

    # Calculate speeds from GPS data
    print("\n[2] Calculating speeds from GPS coordinates...")
    speeds = calculate_speeds(gps_points, timestamps)
    print(f"    Calculated {len(speeds)} speed samples")

    # Analyze performance
    print("\n[3] Analyzing performance metrics...")
    metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Max Speed:        {metrics['max_speed']:.2f} m/s")
    print(f"Min Speed:        {metrics['min_speed']:.2f} m/s")
    print(f"Average Speed:    {metrics['avg_speed']:.2f} m/s")
    print(f"Distance:         {metrics['distance']:.1f} m")
    print(f"Duration:         {metrics['duration']:.0f} seconds ({metrics['duration']/60:.1f} minutes)")
    print(f"\nBattery Voltage:  {metrics['min_voltage']:.2f}V - {metrics['max_voltage']:.2f}V")
    print(f"Max Current:      {metrics['max_current']:.2f} A")
    print(f"Average Current:  {metrics['avg_current']:.2f} A")
    print(f"\nStart Position:   {gps_points[0]}")
    print(f"End Position:     {gps_points[-1]}")
    print("=" * 70)

    # Train ML model on simulated data
    model = train_speed_model(speeds, battery_voltage, motor_current, timestamps)
    features, targets, _ = prepare_training_data(speeds, battery_voltage, motor_current, timestamps)
    regression_metrics = evaluate_model(model, features, targets)
    print("\nREGRESSION METRICS")
    for key, value in regression_metrics.items():
        print(f"  {key}: {value:.5f}")

    # Create visualizations
    print("\n[4] Creating speed vs time plot...")
    plt1 = plot_speed_vs_time(
        speeds,
        timestamps,
        title="Solar Boat Speed vs Time"
    )
    plt1.savefig('solar_boat_speed.png', dpi=150, bbox_inches='tight')
    print("    Saved: solar_boat_speed.png")

    print("\n[5] Creating comprehensive telemetry dashboard...")
    fig = plot_all_metrics(
        speeds,
        battery_voltage,
        motor_current,
        timestamps,
        gps_points
    )
    fig.savefig('solar_boat_telemetry_dashboard.png', dpi=150, bbox_inches='tight')
    print("    Saved: solar_boat_telemetry_dashboard.png")

    print("\n[6] Creating speed plot with GPS coordinates...")
    fig2, ax = plot_with_coordinates(
        speeds,
        timestamps,
        gps_points,
        title="Speed vs Time with GPS Data"
    )
    fig2.savefig('solar_boat_speed_with_gps.png', dpi=150, bbox_inches='tight')
    print("    Saved: solar_boat_speed_with_gps.png")

    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print("Files saved:")
    print("  - solar_boat_speed.png")
    print("  - solar_boat_telemetry_dashboard.png")
    print("  - solar_boat_speed_with_gps.png")
    print("=" * 70)

    # Display plots
    try:
        plt.show()
    except Exception:
        print("\nNote: Plots were saved to files. Display skipped in headless mode.")
