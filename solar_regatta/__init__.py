"""
Solar Regatta Telemetry Analysis Package

Modern tools for analyzing and modeling solar boat race telemetry data.
"""

__version__ = "0.1.0"
__author__ = "Charlie Cullen"

from .core.analysis import (
    calculate_speeds,
    generate_sample_vesc_data,
    analyze_performance,
    plot_speed_vs_time,
    plot_with_coordinates,
    plot_all_metrics,
    dist,
)
from .ml import (
    PerformanceModel,
    evaluate_model,
    forecast_speed_curve,
    prepare_training_data,
    train_on_raw_gps,
    train_speed_model,
)
from .viz import (
    create_current_plot,
    create_efficiency_plot,
    create_gps_path_plot,
    create_speed_plot,
    create_voltage_plot,
)

__all__ = [
    "calculate_speeds",
    "generate_sample_vesc_data",
    "analyze_performance",
    "plot_speed_vs_time",
    "plot_with_coordinates",
    "plot_all_metrics",
    "dist",
    "PerformanceModel",
    "evaluate_model",
    "forecast_speed_curve",
    "prepare_training_data",
    "train_on_raw_gps",
    "train_speed_model",
    "create_current_plot",
    "create_efficiency_plot",
    "create_gps_path_plot",
    "create_speed_plot",
    "create_voltage_plot",
]
