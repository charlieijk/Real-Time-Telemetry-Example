"""
Example: How to plot speed vs time with GPS coordinates from VESC data
"""

from datetime import datetime

from solar_regatta import calculate_speeds, plot_speed_vs_time, plot_with_coordinates

# Example 1: Using MGRS coordinates with timestamps
# Replace these with your actual VESC GPS data
gps_points = [
    "10SEG1234567890",  # MGRS format examples
    "10SEG1234567891",
    "10SEG1234567892",
    "10SEG1234567893",
]

timestamps = [
    datetime(2025, 10, 30, 10, 0, 0),
    datetime(2025, 10, 30, 10, 0, 10),
    datetime(2025, 10, 30, 10, 0, 20),
    datetime(2025, 10, 30, 10, 0, 30),
]

# Calculate speeds between consecutive points
speeds = calculate_speeds(gps_points, timestamps)

# Plot 1: Simple speed vs time
print("Plotting speed vs time...")
plt = plot_speed_vs_time(speeds, timestamps, title="Solar Boat Speed vs Time")
plt.show()

# Plot 2: Speed vs time with GPS coordinates
print("Plotting with GPS coordinates...")
fig, ax = plot_with_coordinates(
    speeds,
    timestamps,
    gps_points,
    title="Solar Boat Speed vs Time (with GPS Data)"
)
fig.savefig('solar_boat_speed.png', dpi=150)  # Save to file
fig.show()

print(f"Max speed: {max(speeds):.2f} m/s" if speeds else "No speeds calculated")
print(f"Average speed: {sum(speeds)/len(speeds):.2f} m/s" if speeds else "No speeds calculated")
