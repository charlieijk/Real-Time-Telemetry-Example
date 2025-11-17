"""ML utilities for Solar Regatta - Advanced ML capabilities."""

# Core models
from .models import (
    PerformanceModel,
    evaluate_model as evaluate_linear_model,
    forecast_speed_curve,
    prepare_training_data,
    train_on_raw_gps,
    train_speed_model,
)

# Feature engineering
from .features import (
    FeatureEngineer,
    create_polynomial_features,
)

# Evaluation
from .evaluation import (
    time_series_split,
    evaluate_model,
    cross_validate,
    compare_models,
)

# Anomaly detection
from .anomaly import (
    SimpleAnomalyDetector,
    detect_voltage_anomalies,
    detect_current_spikes,
    detect_gps_anomalies,
)

# Model inspection
from .model_info import (
    count_model_parameters,
    print_parameter_table,
    get_model_summary,
)

# Tree-based models (optional imports)
try:
    from .tree_models import (
        RandomForestSpeedModel,
        GradientBoostingSpeedModel,
        XGBoostSpeedModel,
        LightGBMSpeedModel,
    )
    TREE_MODELS_AVAILABLE = True
except (ImportError, Exception) as e:
    # Catch all exceptions including XGBoost library loading errors
    TREE_MODELS_AVAILABLE = False
    # print(f"Tree models not available: {e}")

__all__ = [
    # Core
    "PerformanceModel",
    "evaluate_linear_model",
    "forecast_speed_curve",
    "prepare_training_data",
    "train_on_raw_gps",
    "train_speed_model",
    # Features
    "FeatureEngineer",
    "create_polynomial_features",
    # Evaluation
    "time_series_split",
    "evaluate_model",
    "cross_validate",
    "compare_models",
    # Anomaly
    "SimpleAnomalyDetector",
    "detect_voltage_anomalies",
    "detect_current_spikes",
    "detect_gps_anomalies",
    # Model inspection
    "count_model_parameters",
    "print_parameter_table",
    "get_model_summary",
]

# Add tree models if available
if TREE_MODELS_AVAILABLE:
    __all__.extend([
        "RandomForestSpeedModel",
        "GradientBoostingSpeedModel",
        "XGBoostSpeedModel",
        "LightGBMSpeedModel",
    ])
