"""
Neural Network Hybrid Model for Solar Boat Racing.

This module implements physics-informed neural networks that combine:
1. First-principles physics models
2. Learned residual corrections
3. Differentiable physics for end-to-end training
4. Uncertainty quantification via ensembles and dropout

The hybrid approach provides:
- Physical interpretability
- Data efficiency (physics provides strong prior)
- Extrapolation capability beyond training data
- Uncertainty estimates for safety-critical decisions
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import json
from pathlib import Path

from .world_model import BoatState, BoatDynamicsModel, PhysicsParameters, WorldModel


@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network components."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = 'tanh'  # 'tanh', 'relu', 'gelu'
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 100

    # Ensemble settings
    n_ensemble: int = 5
    use_mc_dropout: bool = True
    n_mc_samples: int = 20


class NeuralNetwork:
    """
    Simple feedforward neural network implemented in numpy.

    For production, replace with PyTorch/JAX implementation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = 'tanh',
        dropout_rate: float = 0.1
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Initialize weights
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weights()

        # Training state
        self.training = False

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * std)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        else:
            return np.tanh(x)

    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of activation function."""
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            return 1 - np.tanh(x) ** 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        self._activations = [x]
        self._pre_activations = []

        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            self._pre_activations.append(z)
            x = self._activate(z)

            # Dropout during training
            if self.training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
                x = x * mask / (1 - self.dropout_rate)

            self._activations.append(x)

        # Output layer (no activation)
        z = x @ self.weights[-1] + self.biases[-1]
        self._pre_activations.append(z)

        return z

    def backward(self, grad_output: np.ndarray, lr: float = 1e-3):
        """Backward pass and weight update (simplified SGD)."""
        grad = grad_output

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            grad_w = self._activations[i].T @ grad
            grad_b = grad.sum(axis=0)

            # Update weights
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

            if i > 0:
                # Gradient w.r.t. input to this layer
                grad = grad @ self.weights[i].T
                grad = grad * self._activate_grad(self._pre_activations[i-1])

    def train_mode(self):
        """Set network to training mode."""
        self.training = True

    def eval_mode(self):
        """Set network to evaluation mode."""
        self.training = False

    def get_parameters(self) -> Dict[str, List]:
        """Get network parameters for saving."""
        return {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }

    def set_parameters(self, params: Dict[str, List]):
        """Load network parameters."""
        self.weights = [np.array(w) for w in params['weights']]
        self.biases = [np.array(b) for b in params['biases']]


class ResidualNetwork(NeuralNetwork):
    """
    Residual network that learns corrections to physics model.

    Output is designed to be added to physics predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        residual_scale: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dims)
        self.residual_scale = residual_scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with scaled residual output."""
        raw_output = super().forward(x)
        return raw_output * self.residual_scale


class PhysicsInformedHybridModel:
    """
    Hybrid model combining physics equations with neural network corrections.

    Architecture:
    prediction = physics_model(state, control) + residual_network(state, control)

    The residual network learns to correct systematic biases in the physics model.
    """

    def __init__(
        self,
        physics_params: Optional[PhysicsParameters] = None,
        config: Optional[NeuralNetworkConfig] = None
    ):
        self.params = physics_params or PhysicsParameters()
        self.config = config or NeuralNetworkConfig()

        # Physics model
        self.physics = BoatDynamicsModel(self.params)

        # Residual networks for different quantities
        # State: [velocity, soc, current, sun_intensity] -> corrections
        self.velocity_residual = ResidualNetwork(
            input_dim=6,  # [v, soc, I_motor, sun, heading, time]
            output_dim=1,  # velocity correction
            hidden_dims=self.config.hidden_dims,
            residual_scale=0.2
        )

        self.soc_residual = ResidualNetwork(
            input_dim=6,
            output_dim=1,  # SOC correction
            hidden_dims=self.config.hidden_dims,
            residual_scale=0.05
        )

        # Training data buffer
        self.training_data: List[Dict] = []

        # Trained flag
        self.is_trained = False

    def _prepare_features(self, state: BoatState, control: Tuple[float, float]) -> np.ndarray:
        """Prepare input features for neural network."""
        motor_current, sun_intensity = control

        features = np.array([
            state.velocity / 10.0,  # Normalized
            state.battery_soc,
            motor_current / 15.0,  # Normalized
            sun_intensity / 1000.0,  # Normalized
            np.sin(state.heading),
            state.time / 600.0  # Normalized to ~10 min race
        ])

        return features

    def predict_step(
        self,
        state: BoatState,
        motor_current: float,
        sun_intensity: float,
        dt: float = 1.0,
        use_residual: bool = True
    ) -> BoatState:
        """
        Predict next state using hybrid model.

        Args:
            state: Current boat state
            motor_current: Motor current command (A)
            sun_intensity: Solar irradiance (W/m²)
            dt: Time step (s)
            use_residual: Whether to apply neural network corrections

        Returns:
            Predicted next state
        """
        # Physics prediction
        physics_prediction = self.physics.step(state, motor_current, sun_intensity, dt)

        if not use_residual or not self.is_trained:
            return physics_prediction

        # Neural network corrections
        features = self._prepare_features(state, (motor_current, sun_intensity))

        velocity_correction = float(self.velocity_residual.forward(features)[0, 0])
        soc_correction = float(self.soc_residual.forward(features)[0, 0])

        # Apply corrections
        corrected_state = BoatState(
            time=physics_prediction.time,
            position=physics_prediction.position,
            velocity=max(0, physics_prediction.velocity + velocity_correction),
            heading=physics_prediction.heading,
            battery_voltage=physics_prediction.battery_voltage,
            battery_soc=np.clip(physics_prediction.battery_soc + soc_correction, 0, 1),
            motor_current=motor_current,
            solar_power=physics_prediction.solar_power
        )

        return corrected_state

    def predict_trajectory(
        self,
        initial_state: BoatState,
        control_sequence: List[Tuple[float, float]],
        dt: float = 1.0
    ) -> List[BoatState]:
        """Predict trajectory using hybrid model."""
        states = [initial_state]
        current_state = initial_state

        for motor_current, sun_intensity in control_sequence:
            next_state = self.predict_step(
                current_state, motor_current, sun_intensity, dt
            )
            states.append(next_state)
            current_state = next_state

        return states

    def add_training_sample(
        self,
        state: BoatState,
        control: Tuple[float, float],
        next_state_observed: BoatState,
        dt: float = 1.0
    ):
        """Add observed transition for training."""
        self.training_data.append({
            'state': state,
            'control': control,
            'next_state': next_state_observed,
            'dt': dt
        })

    def train(self, epochs: Optional[int] = None, verbose: bool = False):
        """
        Train residual networks on collected data.

        Minimizes: ||observed - (physics + residual)||²
        """
        if len(self.training_data) < 10:
            return

        epochs = epochs or self.config.n_epochs

        # Prepare training data
        X = []
        y_velocity = []
        y_soc = []

        for sample in self.training_data:
            state = sample['state']
            control = sample['control']
            next_state = sample['next_state']
            dt = sample['dt']

            # Get physics prediction
            physics_pred = self.physics.step(
                state, control[0], control[1], dt
            )

            # Compute residuals (what the NN should learn)
            velocity_residual = next_state.velocity - physics_pred.velocity
            soc_residual = next_state.battery_soc - physics_pred.battery_soc

            features = self._prepare_features(state, control)
            X.append(features)
            y_velocity.append(velocity_residual)
            y_soc.append(soc_residual)

        X = np.array(X)
        y_velocity = np.array(y_velocity).reshape(-1, 1)
        y_soc = np.array(y_soc).reshape(-1, 1)

        # Train networks
        self.velocity_residual.train_mode()
        self.soc_residual.train_mode()

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_v_shuffled = y_velocity[indices]
            y_s_shuffled = y_soc[indices]

            # Mini-batch training
            batch_size = min(self.config.batch_size, len(X))
            total_loss_v = 0.0
            total_loss_s = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y_v = y_v_shuffled[i:i+batch_size]
                batch_y_s = y_s_shuffled[i:i+batch_size]

                # Forward pass
                pred_v = self.velocity_residual.forward(batch_X)
                pred_s = self.soc_residual.forward(batch_X)

                # Compute loss and gradients
                loss_v = np.mean((pred_v - batch_y_v) ** 2)
                loss_s = np.mean((pred_s - batch_y_s) ** 2)

                grad_v = 2 * (pred_v - batch_y_v) / len(batch_X)
                grad_s = 2 * (pred_s - batch_y_s) / len(batch_X)

                # Backward pass
                self.velocity_residual.backward(grad_v, self.config.learning_rate)
                self.soc_residual.backward(grad_s, self.config.learning_rate)

                total_loss_v += loss_v
                total_loss_s += loss_s

            if verbose and epoch % 10 == 0:
                n_batches = len(X) // batch_size + 1
                print(f"Epoch {epoch}: v_loss={total_loss_v/n_batches:.6f}, soc_loss={total_loss_s/n_batches:.6f}")

        self.velocity_residual.eval_mode()
        self.soc_residual.eval_mode()
        self.is_trained = True

    def predict_with_uncertainty(
        self,
        state: BoatState,
        control: Tuple[float, float],
        dt: float = 1.0,
        n_samples: int = None
    ) -> Tuple[BoatState, Dict[str, float]]:
        """
        Predict with uncertainty using MC dropout.

        Returns:
            mean_prediction: Mean predicted state
            uncertainty: Standard deviations for each quantity
        """
        n_samples = n_samples or self.config.n_mc_samples

        if not self.is_trained:
            # No uncertainty if not trained
            pred = self.predict_step(state, control[0], control[1], dt, use_residual=False)
            return pred, {'velocity_std': 0.0, 'soc_std': 0.0}

        # Enable dropout for MC sampling
        self.velocity_residual.train_mode()
        self.soc_residual.train_mode()

        velocities = []
        socs = []

        for _ in range(n_samples):
            pred = self.predict_step(state, control[0], control[1], dt, use_residual=True)
            velocities.append(pred.velocity)
            socs.append(pred.battery_soc)

        self.velocity_residual.eval_mode()
        self.soc_residual.eval_mode()

        mean_velocity = np.mean(velocities)
        mean_soc = np.mean(socs)

        # Get physics prediction for position
        physics_pred = self.physics.step(state, control[0], control[1], dt)

        mean_state = BoatState(
            time=physics_pred.time,
            position=physics_pred.position,
            velocity=mean_velocity,
            heading=physics_pred.heading,
            battery_voltage=physics_pred.battery_voltage,
            battery_soc=mean_soc,
            motor_current=control[0],
            solar_power=physics_pred.solar_power
        )

        uncertainty = {
            'velocity_std': np.std(velocities),
            'soc_std': np.std(socs)
        }

        return mean_state, uncertainty

    def save(self, path: str):
        """Save hybrid model to file."""
        data = {
            'physics_params': {
                'mass': self.params.mass,
                'hull_drag_coeff': self.params.hull_drag_coeff,
                'motor_efficiency': self.params.motor_efficiency,
                'prop_efficiency': self.params.prop_efficiency,
                'battery_capacity': self.params.battery_capacity,
            },
            'velocity_network': self.velocity_residual.get_parameters(),
            'soc_network': self.soc_residual.get_parameters(),
            'is_trained': self.is_trained
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load hybrid model from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Update physics params
        for key, value in data['physics_params'].items():
            setattr(self.params, key, value)
        self.physics = BoatDynamicsModel(self.params)

        # Load networks
        self.velocity_residual.set_parameters(data['velocity_network'])
        self.soc_residual.set_parameters(data['soc_network'])
        self.is_trained = data['is_trained']


class EnsembleHybridModel:
    """
    Ensemble of hybrid models for robust uncertainty quantification.

    Each ensemble member is trained on a bootstrap sample of data.
    """

    def __init__(
        self,
        physics_params: Optional[PhysicsParameters] = None,
        n_ensemble: int = 5,
        config: Optional[NeuralNetworkConfig] = None
    ):
        self.n_ensemble = n_ensemble
        self.config = config or NeuralNetworkConfig()

        # Create ensemble members
        self.models = [
            PhysicsInformedHybridModel(physics_params, self.config)
            for _ in range(n_ensemble)
        ]

        self.training_data: List[Dict] = []

    def add_training_sample(
        self,
        state: BoatState,
        control: Tuple[float, float],
        next_state_observed: BoatState,
        dt: float = 1.0
    ):
        """Add sample to all ensemble members."""
        self.training_data.append({
            'state': state,
            'control': control,
            'next_state': next_state_observed,
            'dt': dt
        })

    def train(self, epochs: Optional[int] = None, verbose: bool = False):
        """Train all ensemble members with bootstrap sampling."""
        n_samples = len(self.training_data)
        epochs = epochs or self.config.n_epochs

        for i, model in enumerate(self.models):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            model.training_data = [self.training_data[j] for j in indices]
            model.train(epochs=epochs, verbose=verbose and i == 0)

    def predict_step(
        self,
        state: BoatState,
        motor_current: float,
        sun_intensity: float,
        dt: float = 1.0
    ) -> Tuple[BoatState, Dict[str, float]]:
        """
        Predict using ensemble with uncertainty quantification.

        Returns:
            mean_state: Ensemble mean prediction
            uncertainty: Ensemble standard deviations
        """
        predictions = []
        for model in self.models:
            pred = model.predict_step(state, motor_current, sun_intensity, dt)
            predictions.append(pred)

        velocities = [p.velocity for p in predictions]
        socs = [p.battery_soc for p in predictions]

        mean_velocity = np.mean(velocities)
        mean_soc = np.mean(socs)

        mean_state = BoatState(
            time=predictions[0].time,
            position=predictions[0].position,  # Use first prediction for position
            velocity=mean_velocity,
            heading=predictions[0].heading,
            battery_voltage=predictions[0].battery_voltage,
            battery_soc=mean_soc,
            motor_current=motor_current,
            solar_power=predictions[0].solar_power
        )

        uncertainty = {
            'velocity_std': np.std(velocities),
            'velocity_epistemic': np.std(velocities),  # Ensemble disagreement
            'soc_std': np.std(socs),
            'soc_epistemic': np.std(socs)
        }

        return mean_state, uncertainty


class DifferentiablePhysicsModel:
    """
    Fully differentiable physics model for end-to-end gradient-based learning.

    Implements boat dynamics with automatic differentiation support.
    All operations are differentiable for backpropagation through physics.
    """

    def __init__(self, params: Optional[PhysicsParameters] = None):
        self.params = params or PhysicsParameters()

        # Learnable physics parameters (can be optimized)
        self.learnable_params = {
            'drag_coeff': self.params.hull_drag_coeff,
            'motor_eff': self.params.motor_efficiency,
            'prop_eff': self.params.prop_efficiency,
            'battery_resistance': self.params.battery_resistance
        }

    def forward(
        self,
        state: np.ndarray,  # [velocity, soc, x, y, heading]
        control: np.ndarray,  # [motor_current, sun_intensity]
        dt: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Differentiable forward pass.

        Args:
            state: Current state vector
            control: Control inputs
            dt: Time step

        Returns:
            next_state: Next state vector
            intermediates: Dictionary of intermediate values for analysis
        """
        velocity = state[0]
        soc = state[1]
        x, y = state[2], state[3]
        heading = state[4]

        motor_current = control[0]
        sun_intensity = control[1]

        # Voltage model
        nominal_voltage = self.params.nominal_voltage
        voltage = nominal_voltage * (0.8 + 0.2 * soc)

        # Power calculations
        voltage_drop = motor_current * self.learnable_params['battery_resistance']
        effective_voltage = voltage - voltage_drop
        motor_power = effective_voltage * motor_current

        # Thrust calculation
        safe_velocity = np.maximum(velocity, 0.1)
        effective_power = motor_power * self.learnable_params['motor_eff'] * self.learnable_params['prop_eff']
        thrust = effective_power / safe_velocity

        # Drag calculation
        drag = (0.5 * self.params.water_density *
                self.learnable_params['drag_coeff'] *
                self.params.frontal_area *
                velocity ** 2)

        # Acceleration and velocity update
        net_force = thrust - drag
        acceleration = net_force / self.params.mass
        new_velocity = np.maximum(0.0, velocity + acceleration * dt)

        # Position update
        displacement = velocity * dt
        new_x = x + displacement * np.cos(heading)
        new_y = y + displacement * np.sin(heading)

        # Solar power
        solar_power = (self.params.solar_panel_area *
                      self.params.solar_efficiency *
                      sun_intensity)

        # Battery update
        net_power = solar_power - motor_power
        energy_change = net_power * dt / 3600.0
        soc_change = energy_change / self.params.battery_capacity
        new_soc = np.clip(soc + soc_change, 0.0, 1.0)

        next_state = np.array([new_velocity, new_soc, new_x, new_y, heading])

        intermediates = {
            'thrust': thrust,
            'drag': drag,
            'acceleration': acceleration,
            'motor_power': motor_power,
            'solar_power': solar_power,
            'net_power': net_power
        }

        return next_state, intermediates

    def gradient(
        self,
        state: np.ndarray,
        control: np.ndarray,
        target: np.ndarray,
        dt: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute gradients of loss w.r.t. learnable parameters.

        Uses finite differences (for numpy; use autograd in production).
        """
        eps = 1e-5
        gradients = {}

        # Base prediction
        pred, _ = self.forward(state, control, dt)
        base_loss = np.sum((pred - target) ** 2)

        for param_name in self.learnable_params:
            # Perturb parameter
            original = self.learnable_params[param_name]
            self.learnable_params[param_name] = original + eps

            pred_perturbed, _ = self.forward(state, control, dt)
            loss_perturbed = np.sum((pred_perturbed - target) ** 2)

            gradients[param_name] = (loss_perturbed - base_loss) / eps

            # Restore
            self.learnable_params[param_name] = original

        return gradients

    def optimize_parameters(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_epochs: int = 100,
        lr: float = 0.001
    ):
        """
        Optimize learnable physics parameters from data.

        Args:
            training_data: List of (state, control, next_state) tuples
            n_epochs: Number of optimization epochs
            lr: Learning rate
        """
        for epoch in range(n_epochs):
            total_loss = 0.0

            for state, control, target in training_data:
                pred, _ = self.forward(state, control)
                loss = np.sum((pred - target) ** 2)
                total_loss += loss

                # Compute gradients
                grads = self.gradient(state, control, target)

                # Update parameters
                for param_name, grad in grads.items():
                    self.learnable_params[param_name] -= lr * grad
                    # Clamp to valid ranges
                    if 'eff' in param_name:
                        self.learnable_params[param_name] = np.clip(
                            self.learnable_params[param_name], 0.3, 0.99
                        )
                    elif 'coeff' in param_name:
                        self.learnable_params[param_name] = np.clip(
                            self.learnable_params[param_name], 0.1, 2.0
                        )


def create_hybrid_model(
    model_type: str = 'standard',
    physics_params: Optional[PhysicsParameters] = None,
    **kwargs
) -> PhysicsInformedHybridModel:
    """
    Factory function to create hybrid model.

    Args:
        model_type: 'standard', 'ensemble', or 'differentiable'
        physics_params: Physics parameters
        **kwargs: Additional configuration

    Returns:
        Configured hybrid model
    """
    if model_type == 'ensemble':
        n_ensemble = kwargs.get('n_ensemble', 5)
        return EnsembleHybridModel(physics_params, n_ensemble)
    elif model_type == 'differentiable':
        return DifferentiablePhysicsModel(physics_params)
    else:
        return PhysicsInformedHybridModel(physics_params)
