"""
Model Predictive Control (MPC) for Real-Time Solar Boat Racing.

This module implements:
1. Receding horizon MPC for optimal motor current control
2. Real-time constraint handling (battery, current limits, safety)
3. Reference tracking for speed/position targets
4. Economic MPC for energy-optimal racing
5. Adaptive MPC with online model updates
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import time

from .world_model import BoatState, BoatDynamicsModel, PhysicsParameters, WorldModel


@dataclass
class MPCConfig:
    """Configuration for Model Predictive Controller."""
    # Horizon parameters
    prediction_horizon: int = 30  # Number of steps to look ahead
    control_horizon: int = 10    # Number of control moves to optimize
    dt: float = 1.0              # Time step (seconds)

    # Weights for cost function
    speed_weight: float = 10.0        # Penalize deviation from target speed
    energy_weight: float = 1.0        # Penalize energy consumption
    smoothness_weight: float = 0.5    # Penalize control changes
    terminal_weight: float = 5.0      # Weight for terminal state

    # Constraints
    min_current: float = 0.0     # Minimum motor current (A)
    max_current: float = 15.0    # Maximum motor current (A)
    min_soc: float = 0.10        # Minimum battery SOC
    max_current_rate: float = 5.0  # Max current change per step (A/s)

    # Solver settings
    solver: str = 'slsqp'        # 'slsqp', 'trust-constr', 'differential_evolution'
    max_iterations: int = 100
    tolerance: float = 1e-4

    # Adaptive settings
    enable_adaptation: bool = True
    adaptation_rate: float = 0.1


@dataclass
class MPCState:
    """Current state for MPC including predictions."""
    boat_state: BoatState
    predicted_trajectory: List[BoatState] = field(default_factory=list)
    optimal_control: np.ndarray = field(default_factory=lambda: np.array([]))
    solve_time_ms: float = 0.0
    cost: float = 0.0
    constraints_satisfied: bool = True
    iteration_count: int = 0


class ModelPredictiveController:
    """
    Real-time Model Predictive Controller for solar boat racing.

    Features:
    - Receding horizon optimization
    - Constraint handling (battery, current limits)
    - Reference tracking (speed, position)
    - Economic MPC mode for energy optimization
    - Warm-starting from previous solution
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: Optional[MPCConfig] = None
    ):
        self.world_model = world_model
        self.config = config or MPCConfig()

        # Warm start - previous solution
        self._prev_solution: Optional[np.ndarray] = None
        self._prev_solve_time: float = 0.0

        # Adaptive model parameters
        self._model_corrections: Dict[str, float] = {
            'drag_factor': 1.0,
            'thrust_factor': 1.0,
            'battery_factor': 1.0
        }

        # Performance tracking
        self._solve_times: List[float] = []
        self._costs: List[float] = []

    def compute_control(
        self,
        current_state: BoatState,
        target_speed: float,
        sun_profile: List[float],
        disturbances: Optional[Dict[str, float]] = None
    ) -> MPCState:
        """
        Compute optimal control action using MPC.

        Args:
            current_state: Current boat state
            target_speed: Desired speed (m/s)
            sun_profile: Predicted solar intensity over horizon
            disturbances: Optional disturbance estimates (wind, current)

        Returns:
            MPCState with optimal control and predictions
        """
        start_time = time.perf_counter()

        # Prepare sun intensity interpolation
        sun_interp = self._prepare_sun_profile(sun_profile)

        # Initial guess (warm start or default)
        x0 = self._get_initial_guess()

        # Define optimization problem
        def objective(u: np.ndarray) -> float:
            return self._compute_cost(
                u, current_state, target_speed, sun_interp, disturbances
            )

        # Constraints
        constraints = self._build_constraints(current_state, sun_interp)

        # Bounds on control inputs
        bounds = [(self.config.min_current, self.config.max_current)] * self.config.control_horizon

        # Solve optimization
        if self.config.solver == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance,
                workers=1
            )
        else:
            result = minimize(
                objective,
                x0,
                method=self.config.solver.upper(),
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tolerance
                }
            )

        solve_time = (time.perf_counter() - start_time) * 1000

        # Extract solution
        optimal_control = result.x if result.success else x0

        # Save for warm start
        self._prev_solution = optimal_control

        # Predict trajectory with optimal control
        control_seq = self._expand_control(optimal_control, sun_interp)
        predicted_trajectory = self.world_model.predict_trajectory(
            current_state, control_seq, self.config.dt
        )

        # Check constraint satisfaction
        constraints_ok = self._check_constraints(predicted_trajectory)

        # Track performance
        self._solve_times.append(solve_time)
        self._costs.append(result.fun if result.success else float('inf'))

        return MPCState(
            boat_state=current_state,
            predicted_trajectory=predicted_trajectory,
            optimal_control=optimal_control,
            solve_time_ms=solve_time,
            cost=result.fun if result.success else float('inf'),
            constraints_satisfied=constraints_ok,
            iteration_count=result.nit if hasattr(result, 'nit') else 0
        )

    def _compute_cost(
        self,
        u: np.ndarray,
        initial_state: BoatState,
        target_speed: float,
        sun_interp: Callable,
        disturbances: Optional[Dict[str, float]]
    ) -> float:
        """Compute MPC cost function."""
        # Expand control to full horizon
        control_seq = self._expand_control(u, sun_interp)

        # Simulate trajectory
        trajectory = self.world_model.predict_trajectory(
            initial_state, control_seq, self.config.dt
        )

        cost = 0.0

        # Speed tracking cost
        for state in trajectory:
            speed_error = (state.velocity - target_speed) ** 2
            cost += self.config.speed_weight * speed_error

        # Energy cost
        for i, state in enumerate(trajectory[1:], 1):
            energy_used = trajectory[i-1].battery_soc - state.battery_soc
            cost += self.config.energy_weight * energy_used * 100

        # Control smoothness cost
        for i in range(1, len(u)):
            delta_u = (u[i] - u[i-1]) ** 2
            cost += self.config.smoothness_weight * delta_u

        # Terminal cost
        if len(trajectory) > 0:
            terminal_speed_error = (trajectory[-1].velocity - target_speed) ** 2
            terminal_soc_penalty = max(0, self.config.min_soc - trajectory[-1].battery_soc) * 1000
            cost += self.config.terminal_weight * (terminal_speed_error + terminal_soc_penalty)

        # Battery constraint violation (soft constraint)
        min_soc = min(s.battery_soc for s in trajectory)
        if min_soc < self.config.min_soc:
            cost += 10000 * (self.config.min_soc - min_soc) ** 2

        return cost

    def _expand_control(
        self,
        u: np.ndarray,
        sun_interp: Callable
    ) -> List[Tuple[float, float]]:
        """Expand control horizon to prediction horizon."""
        control_seq = []

        for i in range(self.config.prediction_horizon):
            if i < len(u):
                current = u[i]
            else:
                # Hold last control value
                current = u[-1] if len(u) > 0 else 0.0

            sun = sun_interp(i * self.config.dt)
            control_seq.append((current, sun))

        return control_seq

    def _prepare_sun_profile(self, sun_profile: List[float]) -> Callable:
        """Create interpolated sun profile."""
        if len(sun_profile) == 0:
            return lambda t: 800.0  # Default

        times = np.arange(len(sun_profile)) * self.config.dt
        return interp1d(
            times, sun_profile,
            kind='linear',
            bounds_error=False,
            fill_value=(sun_profile[0], sun_profile[-1])
        )

    def _get_initial_guess(self) -> np.ndarray:
        """Get initial guess for optimization (warm start)."""
        if self._prev_solution is not None and len(self._prev_solution) == self.config.control_horizon:
            # Shift previous solution
            x0 = np.zeros(self.config.control_horizon)
            x0[:-1] = self._prev_solution[1:]
            x0[-1] = self._prev_solution[-1]
            return x0

        # Default: moderate current
        return np.ones(self.config.control_horizon) * 5.0

    def _build_constraints(
        self,
        initial_state: BoatState,
        sun_interp: Callable
    ) -> List[Dict]:
        """Build constraint functions for optimizer."""
        constraints = []

        # Rate constraint on control changes
        def rate_constraint(u):
            violations = []
            for i in range(1, len(u)):
                rate = abs(u[i] - u[i-1]) / self.config.dt
                violations.append(self.config.max_current_rate - rate)
            return min(violations) if violations else 0.0

        constraints.append({
            'type': 'ineq',
            'fun': rate_constraint
        })

        return constraints

    def _check_constraints(self, trajectory: List[BoatState]) -> bool:
        """Check if trajectory satisfies all constraints."""
        for state in trajectory:
            if state.battery_soc < self.config.min_soc:
                return False
            if state.motor_current < self.config.min_current:
                return False
            if state.motor_current > self.config.max_current:
                return False
        return True

    def adapt_model(
        self,
        measured_state: BoatState,
        predicted_state: BoatState
    ):
        """
        Online model adaptation based on prediction errors.

        Updates internal correction factors to improve predictions.
        """
        if not self.config.enable_adaptation:
            return

        # Compute prediction errors
        velocity_error = measured_state.velocity - predicted_state.velocity
        soc_error = measured_state.battery_soc - predicted_state.battery_soc

        # Update correction factors
        alpha = self.config.adaptation_rate

        if abs(velocity_error) > 0.1:
            # Adjust drag or thrust factor
            if velocity_error > 0:
                # Model under-predicts velocity -> reduce drag or increase thrust
                self._model_corrections['drag_factor'] *= (1.0 - alpha * 0.1)
            else:
                self._model_corrections['drag_factor'] *= (1.0 + alpha * 0.1)

        if abs(soc_error) > 0.01:
            # Adjust battery factor
            if soc_error > 0:
                # Model under-predicts SOC -> reduce consumption
                self._model_corrections['battery_factor'] *= (1.0 - alpha * 0.05)
            else:
                self._model_corrections['battery_factor'] *= (1.0 + alpha * 0.05)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get MPC performance diagnostics."""
        return {
            'avg_solve_time_ms': np.mean(self._solve_times) if self._solve_times else 0.0,
            'max_solve_time_ms': np.max(self._solve_times) if self._solve_times else 0.0,
            'avg_cost': np.mean(self._costs) if self._costs else 0.0,
            'model_corrections': self._model_corrections.copy(),
            'n_solves': len(self._solve_times)
        }

    def reset(self):
        """Reset controller state."""
        self._prev_solution = None
        self._solve_times = []
        self._costs = []


class EconomicMPC(ModelPredictiveController):
    """
    Economic MPC variant optimizing for race completion time and energy.

    Instead of tracking a reference, directly optimizes economic objectives:
    - Minimize race time
    - Maximize remaining battery
    - Optimal speed profile for race
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: Optional[MPCConfig] = None,
        race_distance: float = 1000.0
    ):
        super().__init__(world_model, config)
        self.race_distance = race_distance
        self.distance_covered = 0.0

    def compute_race_control(
        self,
        current_state: BoatState,
        remaining_distance: float,
        sun_profile: List[float],
        competitors_ahead: int = 0
    ) -> MPCState:
        """
        Compute control optimizing for race performance.

        Args:
            current_state: Current boat state
            remaining_distance: Distance left to finish (m)
            sun_profile: Predicted solar intensity
            competitors_ahead: Number of boats ahead (for strategy)
        """
        start_time = time.perf_counter()

        sun_interp = self._prepare_sun_profile(sun_profile)
        x0 = self._get_initial_guess()

        def race_objective(u: np.ndarray) -> float:
            return self._compute_race_cost(
                u, current_state, remaining_distance, sun_interp, competitors_ahead
            )

        bounds = [(self.config.min_current, self.config.max_current)] * self.config.control_horizon

        result = minimize(
            race_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.config.max_iterations}
        )

        solve_time = (time.perf_counter() - start_time) * 1000
        optimal_control = result.x if result.success else x0
        self._prev_solution = optimal_control

        control_seq = self._expand_control(optimal_control, sun_interp)
        predicted_trajectory = self.world_model.predict_trajectory(
            current_state, control_seq, self.config.dt
        )

        return MPCState(
            boat_state=current_state,
            predicted_trajectory=predicted_trajectory,
            optimal_control=optimal_control,
            solve_time_ms=solve_time,
            cost=result.fun if result.success else float('inf'),
            constraints_satisfied=self._check_constraints(predicted_trajectory),
            iteration_count=result.nit if hasattr(result, 'nit') else 0
        )

    def _compute_race_cost(
        self,
        u: np.ndarray,
        initial_state: BoatState,
        remaining_distance: float,
        sun_interp: Callable,
        competitors_ahead: int
    ) -> float:
        """Economic cost function for racing."""
        control_seq = self._expand_control(u, sun_interp)
        trajectory = self.world_model.predict_trajectory(
            initial_state, control_seq, self.config.dt
        )

        # Calculate distance covered in prediction
        distance = 0.0
        for i in range(1, len(trajectory)):
            distance += np.linalg.norm(
                trajectory[i].position - trajectory[i-1].position
            )

        # Time to cover remaining distance (estimated)
        avg_speed = np.mean([s.velocity for s in trajectory])
        if avg_speed > 0.1:
            estimated_time = remaining_distance / avg_speed
        else:
            estimated_time = float('inf')

        # Cost components
        time_cost = estimated_time * 10.0  # Minimize time

        # Battery reserve cost (want to finish with some reserve)
        final_soc = trajectory[-1].battery_soc
        battery_cost = max(0, 0.2 - final_soc) * 1000  # Penalty below 20%

        # Speed profile smoothness
        smoothness_cost = 0.0
        for i in range(1, len(u)):
            smoothness_cost += abs(u[i] - u[i-1]) * self.config.smoothness_weight

        # Aggressive strategy if behind
        if competitors_ahead > 0:
            # Bonus for higher speed
            time_cost *= (1.0 + 0.1 * competitors_ahead)

        return time_cost + battery_cost + smoothness_cost


class AdaptiveMPC(ModelPredictiveController):
    """
    Adaptive MPC that learns from real-time data.

    Uses recursive least squares to update model parameters online.
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: Optional[MPCConfig] = None
    ):
        super().__init__(world_model, config)

        # Recursive estimation state
        self._param_estimates = np.array([
            world_model.params.hull_drag_coeff,
            world_model.params.motor_efficiency,
            world_model.params.prop_efficiency
        ])
        self._covariance = np.eye(3) * 0.1

        # Observation history
        self._observations: List[Tuple[BoatState, float]] = []

    def update_model(
        self,
        measured_state: BoatState,
        applied_control: float
    ):
        """
        Update model parameters using recursive least squares.

        Args:
            measured_state: Observed boat state
            applied_control: Control input that was applied
        """
        self._observations.append((measured_state, applied_control))

        if len(self._observations) < 2:
            return

        # Get previous state
        prev_state, prev_control = self._observations[-2]

        # Predict with current parameters
        predicted = self.world_model.dynamics.step(
            prev_state, prev_control, 800.0, self.config.dt
        )

        # Compute error
        velocity_error = measured_state.velocity - predicted.velocity

        # Simple gradient update for drag coefficient
        if abs(velocity_error) > 0.05:
            learning_rate = 0.01
            # If actual velocity is higher, reduce drag estimate
            self.world_model.params.hull_drag_coeff *= (
                1.0 - learning_rate * np.sign(velocity_error)
            )
            # Clamp to reasonable range
            self.world_model.params.hull_drag_coeff = np.clip(
                self.world_model.params.hull_drag_coeff, 0.1, 2.0
            )


def create_mpc_controller(
    world_model: Optional[WorldModel] = None,
    mode: str = 'standard'
) -> ModelPredictiveController:
    """
    Factory function to create MPC controller.

    Args:
        world_model: World model to use (creates default if None)
        mode: 'standard', 'economic', or 'adaptive'

    Returns:
        Configured MPC controller
    """
    if world_model is None:
        from .world_model import create_default_world_model
        world_model = create_default_world_model()

    config = MPCConfig()

    if mode == 'economic':
        return EconomicMPC(world_model, config)
    elif mode == 'adaptive':
        return AdaptiveMPC(world_model, config)
    else:
        return ModelPredictiveController(world_model, config)
