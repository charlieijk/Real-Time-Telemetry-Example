---
title: Solar Regatta - ML Telemetry Analysis
emoji: 🚤
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
tags:
- machine-learning
- telemetry
- time-series
- solar-boat
- jupyter
- data-analysis
---

# Solar Regatta - Real-Time Telemetry Analysis

A comprehensive Python package for analyzing, modeling, and visualizing solar boat race telemetry data with advanced machine learning capabilities.

## 🎯 What is this?

This project provides a complete toolkit for analyzing telemetry data from solar-powered boats, featuring:

- **Advanced ML Models**: Random Forest, XGBoost, LightGBM for speed prediction
- **Feature Engineering**: 60+ engineered features from raw telemetry
- **Anomaly Detection**: Statistical and domain-specific anomaly detection
- **Interactive Notebooks**: Jupyter notebooks for hands-on learning
- **Real-time Analysis**: GPS tracking, battery monitoring, motor current analysis
- **VESC Integration**: Data collection from VESC motor controllers

## 🚀 Quick Start

### Installation

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/solar-regatta
cd solar-regatta
pip install -e .
```

For advanced ML features:
```bash
pip install -e ".[ml-advanced]"
```

For VESC integration:
```bash
pip install -e ".[vesc]"
```

### Run Example

```bash
# Simple example
python example_vesc_plot.py --mode simple

# Full analysis
python example_vesc_plot.py --mode full

# Without saving plots
python example_vesc_plot.py --no-save-plots
```

### Try the Notebooks

```bash
jupyter lab notebooks/
```

Two comprehensive notebooks available:
- **Advanced_ML_Tutorial.ipynb** - Complete ML pipeline with feature engineering
- **Anomaly_Detection.ipynb** - Telemetry anomaly detection workflows

## 📊 Features

### Machine Learning Capabilities

#### Feature Engineering
- **Rolling Statistics**: Mean, std, max, min over configurable windows
- **Lag Features**: Historical values for temporal context
- **Derivatives**: Speed changes, acceleration patterns
- **Physics Features**: Power, efficiency, energy consumption

#### Models
- **Linear Regression**: Lightweight baseline (NumPy-only)
- **Random Forest**: Ensemble tree model
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting

#### Evaluation
- **Time-series CV**: Respects temporal ordering
- **Comprehensive Metrics**: MSE, RMSE, MAE, R², MAPE, Max Error
- **Model Comparison**: Automated comparison across models
- **Feature Importance**: Understanding what drives predictions

### Anomaly Detection

- **Voltage Anomalies**: Low battery, overvoltage, rapid changes
- **Current Spikes**: Abnormal motor current patterns
- **GPS Anomalies**: Impossible speeds, acceleration limits
- **Statistical Outliers**: Z-score and IQR methods

### VESC Integration

- **Data Collection**: Real-time telemetry from VESC motor controllers
- **Annotation Tools**: Label sessions for ML training
- **Serial Interface**: USB/serial connection support

## 🎓 Use Cases

1. **Racing Performance**: Analyze race telemetry to optimize strategy
2. **Battery Management**: Monitor battery health and predict range
3. **Motor Efficiency**: Understand motor performance patterns
4. **Predictive Maintenance**: Detect anomalies before failures
5. **ML Education**: Learn time-series ML with real-world data

## 📈 Example Usage

### Basic Analysis

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance
)

# Generate sample data
gps, timestamps, speeds, voltage, current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds from GPS
speeds = calculate_speeds(gps, timestamps)

# Get performance metrics
metrics = analyze_performance(speeds, voltage, current, timestamps)

print(f"Max Speed: {metrics['max_speed']:.2f} m/s")
print(f"Distance: {metrics['distance']:.1f} m")
```

### Advanced ML Pipeline

```python
from solar_regatta.ml import (
    FeatureEngineer,
    XGBoostSpeedModel,
    evaluate_model,
    cross_validate
)

# Create feature engineer
engineer = FeatureEngineer(
    rolling_windows=[3, 5, 10],
    lag_features=3,
    include_derivatives=True,
    include_physics=True
)

# Engineer features
X = engineer.fit_transform(speeds, voltages, currents)
y = speeds[engineer.feature_delay:]

# Train XGBoost model
model = XGBoostSpeedModel(n_estimators=100)
model.fit(X, y)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Anomaly Detection

```python
from solar_regatta.ml import (
    detect_voltage_anomalies,
    detect_current_spikes,
    SimpleAnomalyDetector
)

# Detect voltage issues
low_v, high_v, rapid = detect_voltage_anomalies(
    voltages,
    low_threshold=10.5,
    high_threshold=14.0
)

# Detect current spikes
spikes = detect_current_spikes(currents, spike_threshold=3.0)

# Statistical anomalies
detector = SimpleAnomalyDetector(method='zscore', threshold=3.0)
detector.fit(speeds)
anomalies = detector.predict(speeds)
```

## 🏗️ Architecture

```
solar_regatta/
├── core/           # Core analysis functions
│   └── analysis.py # GPS, speed calculations, metrics
├── ml/             # Machine learning
│   ├── models.py       # Linear regression baseline
│   ├── features.py     # Feature engineering
│   ├── tree_models.py  # RF, XGBoost, LightGBM
│   ├── evaluation.py   # CV, metrics, comparison
│   └── anomaly.py      # Anomaly detection
├── vesc/           # VESC integration
│   └── collector.py    # Data collection
└── viz/            # Visualization
    └── plotly_charts.py
```

## 📚 Notebooks

### Advanced_ML_Tutorial.ipynb
Step-by-step tutorial covering:
1. Data generation and preprocessing
2. Feature engineering (60+ features)
3. Model training (Linear, RF, XGBoost)
4. Feature importance analysis
5. Cross-validation
6. Prediction visualization

### Anomaly_Detection.ipynb
Complete anomaly detection guide:
1. Voltage anomaly detection
2. Current spike detection
3. GPS anomaly detection
4. Statistical outlier detection
5. Combined reporting

## 🛠️ Requirements

**Core:**
- Python 3.8+
- NumPy >= 1.24.3
- Matplotlib >= 3.8.0
- Plotly >= 5.17.0
- mgrs >= 1.4.6
- pyproj >= 3.6.0

**Advanced ML (optional):**
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0

**VESC (optional):**
- pyserial >= 3.5

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open issues or submit pull requests.

## 👨‍💻 Author

- Charlie Cullen ([@charlieijk](https://github.com/charlieijk))

## 🙏 Acknowledgments

- Pixel Club SF for the solar boat racing initiative
- VESC community for motor controller telemetry
- Plotly and NumPy communities

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{solar_regatta,
  title = {Solar Regatta: ML Telemetry Analysis for Solar Boats},
  author = {Cullen, Charlie},
  year = {2025},
  url = {https://huggingface.co/spaces/YOUR_USERNAME/solar-regatta}
}
```

---

**Get started now!**
```bash
pip install -e ".[all]"
jupyter lab notebooks/
```
