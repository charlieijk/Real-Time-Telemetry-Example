# Solar Regatta - Real-Time Telemetry Analysis

A comprehensive Python package for analyzing, modeling, and visualizing solar boat race telemetry data. Features machine-learning friendly helpers, interactive dashboards for notebooks, real-time performance metrics, GPS tracking, and VESC motor controller analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

🚤 **Real-Time Telemetry Visualization**
- Interactive dashboard with live data updates
- Speed, voltage, current, and efficiency charts
- GPS track visualization and path analysis

📊 **Performance Analytics**
- Speed calculations from GPS coordinates
- Battery voltage monitoring with cutoff warnings
- Motor current analysis and efficiency metrics
- Comprehensive performance statistics

🌐 **Interactive Visualizations**
- Plotly helpers that render directly in notebooks
- Works offline—perfect for quick experimentation
- Ready-to-use Jupyter notebooks
- Export-ready chart objects for custom dashboards

🐍 **Python Library**
- Easy-to-use API for data analysis and modeling
- Sample data generation for testing
- Matplotlib and Plotly visualizations
- CLI workflow for generating telemetry and training models

## Quick Start

### Installation

**Option 1: Install from GitHub**
```bash
git clone https://github.com/pixelclubsf/Real-Time-Telemetry-Example.git
cd Real-Time-Telemetry-Example
pip install -e .
```

**Option 2: Install normally (no development mode)**
```bash
pip install .
```

### Usage

#### As a Python Library

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance,
    plot_all_metrics
)

# Generate sample telemetry data
gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds from GPS coordinates
speeds = calculate_speeds(gps_points, timestamps)

# Analyze performance metrics
metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

# Create visualizations
plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points)
```

#### Build a Lightweight ML Model

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    train_speed_model,
    prepare_training_data,
    evaluate_model,
)

gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)
speeds = calculate_speeds(gps_points, timestamps)
model = train_speed_model(speeds, battery_voltage, motor_current, timestamps)
X, y, _ = prepare_training_data(speeds, battery_voltage, motor_current, timestamps)
evaluate_model(model, X, y)
```

#### Train a Predictive Model via CLI

Use the included command-line tool to simulate telemetry, fit a regression model, and export the coefficients:

```bash
solar-regatta --duration 600 --interval 5 --save-model model.json --export-predictions predictions.json
```

The CLI prints summary metrics and stores the learned weights in a portable JSON file.

### Example Script

```python
from solar_regatta import calculate_speeds, plot_speed_vs_time
from datetime import datetime, timedelta

# Your VESC GPS data in MGRS format
gps_points = [
    "10SEG1234567890",
    "10SEG1234567891",
    "10SEG1234567892",
]

timestamps = [
    datetime(2025, 10, 30, 10, 0, 0),
    datetime(2025, 10, 30, 10, 0, 10),
    datetime(2025, 10, 30, 10, 0, 20),
]

# Calculate and plot speeds
speeds = calculate_speeds(gps_points, timestamps)
plot_speed_vs_time(speeds, timestamps, title="Solar Boat Speed Analysis")
```

Run the example:
```bash
python example_vesc_plot.py
```

### Jupyter Notebooks

Interactive notebooks are available in the `notebooks/` directory:

- `Solar_Regatta_Quickstart.ipynb` – walk through sample data generation, analysis, and Matplotlib visualizations.
- `Solar_Regatta_Plotly_Dashboard.ipynb` – render the Plotly figures used by the dashboards directly inside Jupyter.

Open them with JupyterLab or VS Code to experiment with live telemetry or tweak the sample data generator.

### Machine Learning Utilities

The `solar_regatta.ml` module provides lightweight linear regression helpers:

- `prepare_training_data` – build feature/target matrices that predict the next speed sample from voltage, current, and timing history.
- `train_speed_model` – fit a regression model (implemented with NumPy’s least-squares solver).
- `evaluate_model` and `forecast_speed_curve` – inspect model quality and produce predictions for your feature matrix.

Combine them with `generate_sample_vesc_data` or your real telemetry feeds to prototype smarter control strategies directly inside notebooks or scripts.

## Core Modules

### `solar_regatta.core.analysis`

**Functions:**
- `calculate_speeds(gps_points, timestamps)` - Calculate speed from GPS coordinates
- `generate_sample_vesc_data(duration_seconds, interval)` - Generate realistic sample data
- `analyze_performance(speeds, battery_voltage, motor_current, timestamps)` - Calculate metrics
- `plot_speed_vs_time(speeds, timestamps, title)` - Create speed visualization
- `plot_with_coordinates(speeds, timestamps, gps_points, title)` - Plot with GPS info
- `plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points)` - Dashboard visualization
- `dist(sp, ep)` - Calculate distance between MGRS coordinates

### `solar_regatta.viz.plotly_charts`

Plotly helpers for notebook dashboards:
- `create_speed_plot(speeds, timestamps)` - Interactive speed vs time chart
- `create_voltage_plot(battery_voltage, timestamps)` - Battery health visualization
- `create_current_plot(motor_current, timestamps)` - Current draw over time
- `create_efficiency_plot(speeds, motor_current)` - Scatter of speed vs current
- `create_gps_path_plot(gps_points)` - Simple sequential GPS path

## API Reference

### Calculate Speeds from GPS

```python
from solar_regatta import calculate_speeds
from datetime import datetime, timedelta

# MGRS coordinates
gps_points = ["10SEG1234567890", "10SEG1234567891"]
# DateTime objects
timestamps = [datetime(2025, 10, 30, 10, 0, 0), datetime(2025, 10, 30, 10, 0, 10)]

# Returns list of speeds in m/s
speeds = calculate_speeds(gps_points, timestamps)
```

### Generate Sample Data

```python
from solar_regatta import generate_sample_vesc_data

gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)
```

**Parameters:**
- `duration_seconds` (int): Total simulation time in seconds (default: 300)
- `interval` (int): Time between GPS samples in seconds (default: 5)

**Returns:**
- `gps_points`: List of MGRS coordinates
- `timestamps`: List of datetime objects
- `speeds_raw`: List of simulated speeds
- `battery_voltage`: List of voltage readings
- `motor_current`: List of current readings

### Analyze Performance

```python
from solar_regatta import analyze_performance

metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

# metrics dict contains:
# - max_speed, min_speed, avg_speed (m/s)
# - max_voltage, min_voltage (V)
# - max_current, avg_current (A)
# - distance (m)
# - duration (seconds)
```

## Data Formats

### GPS Coordinates

The package uses **MGRS (Military Grid Reference System)** format for GPS coordinates:
- Format: `10SEG6400050000` (15 characters)
- Example: `"10SEG1234567890"`
- Converted internally to lat/lon for distance calculations

### Timestamps

Supports both:
- Python `datetime` objects
- Unix timestamps (seconds)

## Modeling & Visualization Workflows

The notebooks and CLI provide:

- **Metrics Display** – Maximum/minimum speed, average speed, current draw, battery voltage windows, total distance, and race duration.
- **Interactive Charts** – Speed vs time, battery voltage, motor current, efficiency scatter plots, and GPS track visualizations using Plotly figures.
- **Model Training** – Quickly fit linear regression models that predict future speeds from voltage/current/elapsed time and export the coefficients as JSON.
- **Data Export** – Save predicted speed curves for downstream use.

## Architecture

```
Solar Regatta
├── Core analysis (solar_regatta/core)
│   ├── GPS distance calculations
│   ├── Speed computations
│   ├── Performance metrics
│   └── Matplotlib visualizations
├── ML utilities (solar_regatta/ml)
│   ├── Feature preparation
│   ├── Linear regression helpers
│   └── Evaluation helpers
├── Plotly visuals (solar_regatta/viz)
│   └── Notebook-friendly chart builders
└── Notebooks & CLI
    ├── Example notebooks in /notebooks
    └── `solar-regatta` command for quick experiments
```

## Requirements

- Python 3.8 or higher
- Plotly 5.17.0
- Matplotlib 3.8.0
- NumPy 1.24.3
- MGRS 1.4.6

All dependencies are installed automatically with `pip install -e .`.

## Installation Troubleshooting

**ModuleNotFoundError: No module named 'solar_regatta'**
```bash
pip install -e .
```

**Plotly or NumPy import errors**
```bash
pip install -r requirements.txt
```

**CLI not found**
```bash
pip install -e .
```

## Project Structure

```
Real-Time-Telemetry-Example/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── example_vesc_plot.py            # Matplotlib example
├── solar.py                        # Stand-alone analysis script
├── notebooks/                      # Interactive workflows
│   ├── Solar_Regatta_Quickstart.ipynb
│   └── Solar_Regatta_Plotly_Dashboard.ipynb
└── solar_regatta/                  # Installable package
    ├── __init__.py
    ├── cli.py                      # Command-line entry point
    ├── core/
    │   └── analysis.py
    ├── ml/
    │   └── models.py
    └── viz/
        └── plotly_charts.py
```

## Usage Examples

### Basic Analysis Script

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance
)

# Generate sample data (300 second race)
gps, timestamps, speeds_raw, voltage, current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds
speeds = calculate_speeds(gps, timestamps)

# Get metrics
metrics = analyze_performance(speeds, voltage, current, timestamps)

print(f"Max Speed: {metrics['max_speed']:.2f} m/s")
print(f"Avg Speed: {metrics['avg_speed']:.2f} m/s")
print(f"Distance: {metrics['distance']:.1f} m")
print(f"Duration: {metrics['duration']:.0f} seconds")
```

### Custom GPS Data

```python
from solar_regatta import calculate_speeds
from datetime import datetime, timedelta

# Your actual VESC GPS data
your_gps_points = ["10SEG...", "10SEG...", ...]
your_timestamps = [datetime(...), datetime(...), ...]

speeds = calculate_speeds(your_gps_points, your_timestamps)
```

### Data Export

```python
import json
from solar_regatta import generate_sample_vesc_data, calculate_speeds

# Generate data
gps, timestamps, speeds_raw, voltage, current = \
    generate_sample_vesc_data()

speeds = calculate_speeds(gps, timestamps)

# Export to JSON
data = {
    'gps_points': gps,
    'speeds': speeds,
    'battery_voltage': voltage,
    'motor_current': current
}

with open('telemetry_data.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- GitHub Issues: [Open an issue](https://github.com/pixelclubsf/Real-Time-Telemetry-Example/issues)
- Documentation: See [FLASK_README.md](FLASK_README.md) for detailed feature docs

## Project Status

**Version:** 0.1.0 (Alpha)
**Status:** Active Development
**Last Updated:** October 2025

## Authors

- Charlie Cullen (@charlieijk)

## Acknowledgments

- Pixel Club SF for the solar boat racing initiative
- VESC (Vedder's ESC) community for motor controller telemetry data
- Plotly for interactive visualization
- NumPy community for dependable scientific tooling

---

**Get started now!**
```bash
pip install -e .
solar-regatta --duration 600 --interval 5 --save-model model.json
```
