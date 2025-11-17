#!/usr/bin/env python3
"""
Solar Regatta - Comprehensive Dashboard
Multi-tab Gradio interface for VESC data collection, model training, inference, and analysis.
Deployed on Hugging Face Spaces.
"""

import gradio as gr
import numpy as np
import json
import io
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Import Solar Regatta modules
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance,
)
from solar_regatta.ml import (
    train_speed_model,
    prepare_training_data,
    count_model_parameters,
    FeatureEngineer,
)
from solar_regatta.ml.models import PerformanceModel
from solar_regatta.viz.plotly_charts import (
    create_speed_plot,
    create_voltage_plot,
    create_current_plot,
    create_efficiency_plot,
    create_gps_path_plot,
)

# Try to import tree models
try:
    from solar_regatta.ml import RandomForestSpeedModel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from solar_regatta.ml import XGBoostSpeedModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from solar_regatta.vesc import VESCDataCollector
    VESC_AVAILABLE = True
except ImportError:
    VESC_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_parameter_table(model, X=None):
    """Format parameter information as HTML table."""
    info = count_model_parameters(model)

    html = f"""
    <div style="font-family: monospace; padding: 20px;">
        <h2>Model Parameter Summary</h2>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Model Type</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{info['model_type']}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total Parameters</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{info['total_parameters']:,}</td>
            </tr>
    """

    if X is not None:
        html += f"""
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Training Samples</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{X.shape[0]:,}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Features</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{X.shape[1]:,}</td>
            </tr>
        """

    html += "</table>"

    # Coefficients table for linear models
    if isinstance(model, PerformanceModel):
        html += """
        <h3>Coefficients</h3>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #4CAF50; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Feature</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Coefficient</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Abs Value</th>
            </tr>
        """

        for name, coef in zip(model.feature_names, model.coefficients):
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{name}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{coef:.6f}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{abs(coef):.6f}</td>
            </tr>
            """

        html += f"""
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Intercept</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><strong>{model.intercept:.6f}</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><strong>{abs(model.intercept):.6f}</strong></td>
            </tr>
        </table>
        """

        # Feature importance
        importance_order = np.argsort(np.abs(model.coefficients))[::-1]
        html += """
        <h3>Feature Importance</h3>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #2196F3; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Rank</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Feature</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Importance</th>
            </tr>
        """

        for rank, idx in enumerate(importance_order, 1):
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{rank}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{model.feature_names[idx]}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{abs(model.coefficients[idx]):.6f}</td>
            </tr>
            """

        html += "</table>"

    html += "</div>"
    return html


def create_feature_importance_plot(model):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(model, PerformanceModel):
        importance = np.abs(model.coefficients)
        features = model.feature_names

        # Sort by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(len(sorted_idx))

        ax.barh(pos, importance[sorted_idx], color='#4CAF50')
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Feature Importance (Linear Model)')
        ax.grid(axis='x', alpha=0.3)

    elif hasattr(model, 'get_feature_importance'):
        try:
            importance = model.get_feature_importance()
            features = [f"Feature {i}" for i in range(len(importance))]

            # Sort by importance
            sorted_idx = np.argsort(importance)[-10:]  # Top 10
            pos = np.arange(len(sorted_idx))

            ax.barh(pos, importance[sorted_idx], color='#2196F3')
            ax.set_yticks(pos)
            ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importances')
            ax.grid(axis='x', alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Feature importance not available',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


# ============================================================================
# TAB 1: VESC DATA COLLECTION
# ============================================================================

def collect_vesc_data_simulation(duration, interval):
    """Simulate VESC data collection (since real hardware may not be available)."""
    try:
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        # Format data for display
        data = []
        for i, (gps_pt, ts, speed, volt, curr) in enumerate(
            zip(gps, timestamps, speeds_raw, voltage, current)
        ):
            data.append({
                "Sample": i + 1,
                "Timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "GPS (MGRS)": gps_pt,
                "Speed (m/s)": f"{speed:.2f}",
                "Voltage (V)": f"{volt:.2f}",
                "Current (A)": f"{curr:.2f}",
            })

        json_data = json.dumps(data, indent=2)

        # Create summary
        summary = f"""
        ### Collection Summary
        - **Duration:** {duration} seconds
        - **Samples Collected:** {len(gps)}
        - **Sampling Interval:** {interval} seconds
        - **Speed Range:** {min(speeds_raw):.2f} - {max(speeds_raw):.2f} m/s
        - **Voltage Range:** {min(voltage):.2f} - {max(voltage):.2f} V
        - **Current Range:** {min(current):.2f} - {max(current):.2f} A
        """

        return json_data, summary
    except Exception as e:
        return f"Error: {str(e)}", f"Error collecting data: {str(e)}"


def export_collection_csv(json_str):
    """Export collection data as CSV."""
    try:
        data = json.loads(json_str)
        csv_lines = ["Sample,Timestamp,GPS (MGRS),Speed (m/s),Voltage (V),Current (A)"]
        for row in data:
            csv_lines.append(
                f"{row['Sample']},{row['Timestamp']},{row['GPS (MGRS)']},{row['Speed (m/s)']},{row['Voltage (V)']},{row['Current (A)']}"
            )
        csv_content = "\n".join(csv_lines)
        return csv_content
    except Exception as e:
        return f"Error exporting: {str(e)}"


# ============================================================================
# TAB 2: MODEL TRAINING
# ============================================================================

def train_and_analyze(model_type, duration, n_samples, use_feature_engineering):
    """Train model and return parameter analysis."""
    try:
        # Generate data
        interval = max(1, duration // n_samples)
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        speeds = calculate_speeds(gps, timestamps)

        # Prepare training data
        if use_feature_engineering:
            engineer = FeatureEngineer(
                rolling_windows=[3, 5],
                lag_features=2,
                include_derivatives=True,
                include_physics=True
            )
            X = engineer.fit_transform(speeds, voltage, current)
            y = speeds[engineer.feature_delay:]

            # Train linear model with engineered features
            X_aug = np.column_stack([X, np.ones(len(X))])
            solution, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            coefficients = solution[:-1]
            intercept = solution[-1]

            model = PerformanceModel(
                coefficients=coefficients,
                intercept=intercept,
                feature_names=engineer.get_feature_names()
            )

            # Calculate R² on training data
            predictions = X @ coefficients + intercept
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            X, y, _ = prepare_training_data(speeds, voltage, current, timestamps)

            if model_type == "Linear Regression":
                model = train_speed_model(speeds, voltage, current, timestamps)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            elif model_type == "Random Forest" and SKLEARN_AVAILABLE:
                model = RandomForestSpeedModel(n_estimators=50, max_depth=10)
                model.fit(X, y)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                model = XGBoostSpeedModel(n_estimators=50, max_depth=6)
                model.fit(X, y, verbose=False)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            else:
                model = train_speed_model(speeds, voltage, current, timestamps)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Generate outputs
        param_table = format_parameter_table(model, X)
        feature_plot = create_feature_importance_plot(model)

        # Summary stats
        info = count_model_parameters(model)
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(np.mean((y - predictions) ** 2))

        summary = f"""
        ### Model Summary
        - **Type:** {info['model_type']}
        - **Total Parameters:** {info['total_parameters']:,}
        - **Training Samples:** {X.shape[0]:,}
        - **Features:** {X.shape[1]:,}
        - **R² Score:** {r2_score:.4f}
        - **MAE:** {mae:.4f} m/s
        - **RMSE:** {rmse:.4f} m/s
        """

        return param_table, feature_plot, summary
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        return error_msg, None, error_msg


# ============================================================================
# TAB 3: SPEED PREDICTION
# ============================================================================

# Global state for saved models
saved_models = {}

def train_inference_model(model_type, duration, n_samples, use_feature_engineering):
    """Train and save model for inference."""
    try:
        interval = max(1, duration // n_samples)
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        speeds = calculate_speeds(gps, timestamps)

        if use_feature_engineering:
            engineer = FeatureEngineer(
                rolling_windows=[3, 5],
                lag_features=2,
                include_derivatives=True,
                include_physics=True
            )
            X = engineer.fit_transform(speeds, voltage, current)
            y = speeds[engineer.feature_delay:]
            X_aug = np.column_stack([X, np.ones(len(X))])
            solution, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            coefficients = solution[:-1]
            intercept = solution[-1]
            model = PerformanceModel(
                coefficients=coefficients,
                intercept=intercept,
                feature_names=engineer.get_feature_names()
            )
        else:
            X, y, _ = prepare_training_data(speeds, voltage, current, timestamps)
            if model_type == "Linear Regression":
                model = train_speed_model(speeds, voltage, current, timestamps)
            elif model_type == "Random Forest" and SKLEARN_AVAILABLE:
                model = RandomForestSpeedModel(n_estimators=50, max_depth=10)
                model.fit(X, y)
            elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                model = XGBoostSpeedModel(n_estimators=50, max_depth=6)
                model.fit(X, y, verbose=False)
            else:
                model = train_speed_model(speeds, voltage, current, timestamps)

        saved_models['current'] = {
            'model': model,
            'type': model_type,
            'engineered': use_feature_engineering
        }

        return "✓ Model trained and ready for predictions!", f"Model Type: {model_type}"
    except Exception as e:
        return f"Error: {str(e)}", "Failed to train model"


def predict_speed(voltage, current, timestamp_str, prev_speed=None):
    """Make speed prediction using saved model."""
    try:
        if 'current' not in saved_models:
            return "No model trained yet! Train a model first.", "N/A"

        model_info = saved_models['current']
        model = model_info['model']

        # Create feature vector
        if prev_speed is None:
            prev_speed = 0.0

        features = np.array([[prev_speed, voltage, current, 1.0]])

        prediction = model.predict(features)[0]

        return f"**Predicted Speed:** {prediction:.3f} m/s", f"Input - Voltage: {voltage}V, Current: {current}A"
    except Exception as e:
        return f"Error: {str(e)}", "Prediction failed"


# ============================================================================
# TAB 4: DATA ANALYSIS DASHBOARD
# ============================================================================

def analyze_uploaded_data(file):
    """Analyze uploaded telemetry data and create visualizations."""
    try:
        if file is None:
            return None, None, None, None, None, "Please upload a telemetry file"

        # Load JSON data
        file_content = file.read().decode('utf-8')
        data = json.loads(file_content)

        # Parse telemetry data
        gps_points = [d.get("gps_position", f"10SEG{i:010d}") for i, d in enumerate(data)]
        speeds = [float(d.get("speed_gps", 0)) for d in data]
        voltages = [float(d.get("battery_voltage", 13)) for d in data]
        currents = [float(d.get("motor_current", 5)) for d in data]

        # Create visualizations
        fig_speed = create_speed_plot(speeds, list(range(len(speeds))))
        fig_voltage = create_voltage_plot(voltages, list(range(len(voltages))))
        fig_current = create_current_plot(currents, list(range(len(currents))))
        fig_efficiency = create_efficiency_plot(speeds, currents)
        fig_gps = create_gps_path_plot(gps_points)

        # Analyze performance
        metrics = analyze_performance(speeds, voltages, currents, list(range(len(speeds))))

        summary = f"""
        ### Telemetry Analysis Summary
        - **Total Samples:** {len(speeds)}
        - **Max Speed:** {metrics['max_speed']:.2f} m/s
        - **Avg Speed:** {metrics['avg_speed']:.2f} m/s
        - **Min Speed:** {metrics['min_speed']:.2f} m/s
        - **Distance:** {metrics['distance']:.1f} m
        - **Duration:** {metrics['duration']:.0f} seconds
        - **Max Voltage:** {metrics['max_voltage']:.2f} V
        - **Min Voltage:** {metrics['min_voltage']:.2f} V
        - **Max Current:** {metrics['max_current']:.2f} A
        - **Avg Current:** {metrics['avg_current']:.2f} A
        """

        return fig_speed, fig_voltage, fig_current, fig_efficiency, fig_gps, summary
    except Exception as e:
        return None, None, None, None, None, f"Error analyzing data: {str(e)}"


# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="Solar Regatta - VESC Dashboard",
    theme=gr.themes.Soft(),
    css="""
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: white; border-radius: 10px; }
    .tab-title { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    """
) as demo:

    gr.Markdown("""
    # 🚤 Solar Regatta - VESC Dashboard

    Comprehensive suite for solar boat telemetry analysis:
    - Collect real-time VESC motor controller data
    - Train machine learning models on telemetry
    - Make real-time speed predictions
    - Analyze and visualize race data

    **Current Status:** Ready for Hugging Face Spaces deployment
    """)

    with gr.Tabs():

        # ====================================================================
        # TAB 1: VESC DATA COLLECTION
        # ====================================================================
        with gr.Tab("📊 Data Collection"):
            gr.Markdown("## VESC Data Collection & Export")
            gr.Markdown("""
            Simulate VESC motor controller telemetry data collection.
            In production, this connects to real VESC hardware via serial port.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    duration_collection = gr.Slider(
                        minimum=30,
                        maximum=600,
                        value=120,
                        step=30,
                        label="Collection Duration (seconds)"
                    )
                    interval_collection = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Sampling Interval (seconds)"
                    )
                    collect_btn = gr.Button("🔴 Start Collection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Live Data Stream")
                    collection_summary = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Raw Data (JSON)")
                    collection_data = gr.Textbox(
                        label="Telemetry Data",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                    export_csv_btn = gr.Button("📥 Export as CSV")
                    csv_output = gr.Textbox(
                        label="CSV Export",
                        lines=5,
                        interactive=False
                    )

            # Connect collection
            collect_btn.click(
                fn=collect_vesc_data_simulation,
                inputs=[duration_collection, interval_collection],
                outputs=[collection_data, collection_summary]
            )

            export_csv_btn.click(
                fn=export_collection_csv,
                inputs=[collection_data],
                outputs=[csv_output]
            )

        # ====================================================================
        # TAB 2: MODEL TRAINING
        # ====================================================================
        with gr.Tab("🤖 Model Training"):
            gr.Markdown("## Train ML Models on VESC Data")
            gr.Markdown("""
            Train various machine learning models to predict solar boat speed
            from battery voltage and motor current.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")

                    model_type_train = gr.Dropdown(
                        choices=["Linear Regression", "Random Forest", "XGBoost"],
                        value="Linear Regression",
                        label="Model Type"
                    )

                    duration_train = gr.Slider(
                        minimum=60,
                        maximum=600,
                        value=300,
                        step=60,
                        label="Simulation Duration (seconds)"
                    )

                    n_samples_train = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=60,
                        step=10,
                        label="Number of Samples"
                    )

                    use_feature_eng_train = gr.Checkbox(
                        label="Use Feature Engineering (20+ features)",
                        value=False
                    )

                    train_model_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Training Summary")
                    summary_output = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Parameter Table")
                    param_table_output = gr.HTML()

                with gr.Column():
                    gr.Markdown("### Feature Importance")
                    feature_plot_output = gr.Image(label="Feature Importance Plot")

            # Connect training
            train_model_btn.click(
                fn=train_and_analyze,
                inputs=[model_type_train, duration_train, n_samples_train, use_feature_eng_train],
                outputs=[param_table_output, feature_plot_output, summary_output]
            )

        # ====================================================================
        # TAB 3: SPEED PREDICTION
        # ====================================================================
        with gr.Tab("⚡ Speed Predictions"):
            gr.Markdown("## Real-Time Speed Prediction")
            gr.Markdown("""
            Train a model and use it to predict boat speed from electrical measurements.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Train Model")

                    model_type_pred = gr.Dropdown(
                        choices=["Linear Regression", "Random Forest", "XGBoost"],
                        value="Linear Regression",
                        label="Model Type"
                    )

                    duration_pred = gr.Slider(
                        minimum=60,
                        maximum=600,
                        value=300,
                        step=60,
                        label="Training Duration (seconds)"
                    )

                    n_samples_pred = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=60,
                        step=10,
                        label="Training Samples"
                    )

                    use_feature_eng_pred = gr.Checkbox(
                        label="Use Feature Engineering",
                        value=False
                    )

                    train_pred_btn = gr.Button("🔧 Prepare Model", variant="primary")
                    train_status = gr.Textbox(label="Training Status", interactive=False)
                    train_info = gr.Textbox(label="Model Info", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Step 2: Make Predictions")

                    voltage_input = gr.Number(
                        value=13.0,
                        label="Battery Voltage (V)"
                    )

                    current_input = gr.Number(
                        value=5.0,
                        label="Motor Current (A)"
                    )

                    prev_speed_input = gr.Number(
                        value=2.0,
                        label="Previous Speed (m/s)"
                    )

                    predict_btn = gr.Button("🎯 Predict Speed", variant="primary")

                    prediction_output = gr.Textbox(label="Prediction Result", interactive=False)
                    prediction_input_info = gr.Textbox(label="Input Summary", interactive=False)

            # Connect prediction
            train_pred_btn.click(
                fn=train_inference_model,
                inputs=[model_type_pred, duration_pred, n_samples_pred, use_feature_eng_pred],
                outputs=[train_status, train_info]
            )

            predict_btn.click(
                fn=predict_speed,
                inputs=[voltage_input, current_input, gr.Textbox(value="2024-01-01", visible=False), prev_speed_input],
                outputs=[prediction_output, prediction_input_info]
            )

        # ====================================================================
        # TAB 4: DATA ANALYSIS
        # ====================================================================
        with gr.Tab("📈 Analysis Dashboard"):
            gr.Markdown("## Telemetry Data Analysis & Visualization")
            gr.Markdown("""
            Upload your race telemetry data and explore interactive visualizations
            of speed, voltage, current, and efficiency metrics.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Data")
                    data_file = gr.File(
                        label="Upload JSON Telemetry",
                        file_types=[".json"]
                    )
                    analyze_btn = gr.Button("📊 Analyze Data", variant="primary", size="lg")
                    analysis_summary = gr.Markdown()

                with gr.Column(scale=2):
                    gr.Markdown("### Quick Stats")

            gr.Markdown("### Visualizations")

            with gr.Row():
                with gr.Column():
                    speed_plot = gr.Plot(label="Speed vs Time")
                with gr.Column():
                    voltage_plot = gr.Plot(label="Voltage vs Time")

            with gr.Row():
                with gr.Column():
                    current_plot = gr.Plot(label="Current vs Time")
                with gr.Column():
                    efficiency_plot = gr.Plot(label="Speed vs Current")

            with gr.Row():
                gps_plot = gr.Plot(label="GPS Track")

            # Connect analysis
            analyze_btn.click(
                fn=analyze_uploaded_data,
                inputs=[data_file],
                outputs=[speed_plot, voltage_plot, current_plot, efficiency_plot, gps_plot, analysis_summary]
            )

        # ====================================================================
        # TAB 5: ABOUT
        # ====================================================================
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            # About Solar Regatta

            **Solar Regatta** is a comprehensive Python toolkit for analyzing telemetry data
            from solar-powered boats competing in autonomous races.

            ## Features

            - **Real-time VESC Integration** - Connect directly to VESC motor controllers
            - **GPS Tracking** - MGRS format GPS position logging and path analysis
            - **ML Models** - Train linear regression, random forest, and XGBoost models
            - **Feature Engineering** - 40+ derived features from raw telemetry
            - **Interactive Visualizations** - Plotly-based charts in notebooks and web UIs
            - **Performance Analytics** - Speed, efficiency, battery, and motor metrics

            ## Quick Links

            - **GitHub:** [charlieijk/SolarRegetta](https://github.com/charlieijk/SolarRegetta)
            - **Hugging Face Space:** [charlieijk/solar-regatta](https://huggingface.co/spaces/charlieijk/solar-regatta)
            - **Documentation:** [VESC Integration Guide](https://github.com/charlieijk/SolarRegetta/blob/main/VESC_INTEGRATION.md)

            ## Technologies

            - Python 3.8+
            - NumPy, SciPy, Scikit-learn
            - Plotly, Matplotlib
            - Gradio for web UI
            - PySerial for VESC communication

            ## Author

            Charlie Cullen (@charlieijk)

            Built with ❤️ for the Pixel Club SF solar boat racing initiative.
            """)

    gr.Markdown("""
    ---

    **Solar Regatta v0.1.0** | Deployed on Hugging Face Spaces

    [GitHub](https://github.com/charlieijk/SolarRegetta) |
    [Issues](https://github.com/charlieijk/SolarRegetta/issues) |
    [Discussion](https://huggingface.co/spaces/charlieijk/solar-regatta/discussions)
    """)


if __name__ == "__main__":
    demo.launch(share=False)
