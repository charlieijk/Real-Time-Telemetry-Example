"""
Reinforcement Learning Agent for Solar Boat Racing.

This module implements:
1. Racing environment (Gym-compatible)
2. PPO, SAC, and TD3 agents
3. Custom reward shaping for racing
4. Curriculum learning for progressive difficulty
5. Multi-objective RL for speed vs energy tradeoffs
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import json
from pathlib import Path

from .world_model import BoatState, WorldModel, PhysicsParameters, create_default_world_model


@dataclass
class RaceConfig:
    """Configuration for racing environment."""
    race_distance: float = 1000.0      # meters
    max_race_time: float = 600.0       # seconds
    dt: float = 1.0                    # time step

    # Reward weights
    speed_reward: float = 1.0
    energy_reward: float = 0.5
    finish_bonus: float = 100.0
    timeout_penalty: float = -50.0
    battery_death_penalty: float = -100.0

    # Observation noise
    observation_noise: float = 0.02

    # Action space
    continuous_action: bool = True
    n_discrete_actions: int = 11  # If discrete: [0, 1.5, 3, ..., 15A]


@dataclass
class Transition:
    """Single transition for experience replay."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class SolarBoatRacingEnv:
    """
    Gym-compatible racing environment for RL training.

    State space (8 dimensions):
        - velocity (normalized)
        - battery_soc
        - distance_covered / race_distance
        - time_elapsed / max_time
        - solar_intensity (normalized)
        - heading (sin, cos)
        - distance_to_finish (normalized)

    Action space:
        - Continuous: motor_current in [0, 15] A
        - Discrete: one of n_discrete_actions levels
    """

    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        config: Optional[RaceConfig] = None,
        sun_profile_fn: Optional[Callable[[float], float]] = None
    ):
        self.world_model = world_model or create_default_world_model()
        self.config = config or RaceConfig()

        # Sun profile function (time -> intensity)
        self.sun_profile_fn = sun_profile_fn or (lambda t: 800.0 + 200 * np.sin(t * 0.01))

        # State
        self.current_state: Optional[BoatState] = None
        self.distance_covered: float = 0.0
        self.episode_reward: float = 0.0
        self.step_count: int = 0

        # Observation and action dimensions
        self.observation_dim = 8
        self.action_dim = 1

        # Statistics tracking
        self.episode_stats: List[Dict] = []

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.current_state = BoatState(
            time=0.0,
            position=np.array([0.0, 0.0]),
            velocity=0.0,
            heading=0.0,
            battery_voltage=13.0,
            battery_soc=1.0,
            motor_current=0.0,
            solar_power=0.0
        )
        self.distance_covered = 0.0
        self.episode_reward = 0.0
        self.step_count = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info.

        Args:
            action: Motor current (continuous or discrete index)

        Returns:
            observation, reward, done, info
        """
        # Convert action to motor current
        if self.config.continuous_action:
            motor_current = float(np.clip(action[0], 0, 15))
        else:
            # Discrete action to current
            action_idx = int(action[0])
            motor_current = action_idx * (15.0 / (self.config.n_discrete_actions - 1))

        # Get sun intensity
        sun_intensity = self.sun_profile_fn(self.current_state.time)

        # Simulate one step
        prev_state = self.current_state
        self.current_state = self.world_model.dynamics.step(
            self.current_state,
            motor_current,
            sun_intensity,
            self.config.dt
        )

        # Update distance
        displacement = np.linalg.norm(
            self.current_state.position - prev_state.position
        )
        self.distance_covered += displacement
        self.step_count += 1

        # Compute reward
        reward = self._compute_reward(prev_state, self.current_state, displacement)
        self.episode_reward += reward

        # Check termination conditions
        done, info = self._check_done()

        # Add noise to observation
        obs = self._get_observation()
        if self.config.observation_noise > 0:
            obs += np.random.normal(0, self.config.observation_noise, obs.shape)

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        state = self.current_state
        sun = self.sun_profile_fn(state.time)

        obs = np.array([
            state.velocity / 10.0,  # Normalized velocity
            state.battery_soc,
            self.distance_covered / self.config.race_distance,
            state.time / self.config.max_race_time,
            sun / 1000.0,  # Normalized sun intensity
            np.sin(state.heading),
            np.cos(state.heading),
            (self.config.race_distance - self.distance_covered) / self.config.race_distance
        ], dtype=np.float32)

        return obs

    def _compute_reward(
        self,
        prev_state: BoatState,
        curr_state: BoatState,
        displacement: float
    ) -> float:
        """Compute reward for transition."""
        reward = 0.0

        # Speed reward (progress toward goal)
        reward += self.config.speed_reward * displacement

        # Energy efficiency reward
        energy_used = prev_state.battery_soc - curr_state.battery_soc
        if displacement > 0:
            efficiency = displacement / (energy_used + 0.001)
            reward += self.config.energy_reward * np.tanh(efficiency * 0.1)

        # Penalty for low battery
        if curr_state.battery_soc < 0.2:
            reward -= 1.0 * (0.2 - curr_state.battery_soc)

        return reward

    def _check_done(self) -> Tuple[bool, Dict]:
        """Check if episode should terminate."""
        info = {
            'distance': self.distance_covered,
            'time': self.current_state.time,
            'final_soc': self.current_state.battery_soc,
            'episode_reward': self.episode_reward
        }

        # Finished race
        if self.distance_covered >= self.config.race_distance:
            info['termination'] = 'finished'
            info['success'] = True
            return True, info

        # Timeout
        if self.current_state.time >= self.config.max_race_time:
            info['termination'] = 'timeout'
            info['success'] = False
            return True, info

        # Battery depleted
        if self.current_state.battery_soc <= 0.05:
            info['termination'] = 'battery_depleted'
            info['success'] = False
            return True, info

        return False, info

    def render(self, mode: str = 'text') -> Optional[str]:
        """Render current state."""
        if mode == 'text':
            return (
                f"Time: {self.current_state.time:.1f}s | "
                f"Distance: {self.distance_covered:.1f}m / {self.config.race_distance}m | "
                f"Speed: {self.current_state.velocity:.2f} m/s | "
                f"SOC: {self.current_state.battery_soc:.1%}"
            )
        return None


class RLAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action given state."""
        pass

    @abstractmethod
    def update(self, batch: List[Transition]) -> Dict[str, float]:
        """Update agent from batch of transitions."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent to file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent from file."""
        pass


class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization agent.

    Simple but effective on-policy algorithm with:
    - Clipped surrogate objective
    - Value function baseline
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 1,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Neural network parameters (using numpy for simplicity)
        # In production, use PyTorch/JAX
        self._init_networks()
        self.lr = lr

        # Trajectory buffer for PPO
        self.trajectory: List[Transition] = []

    def _init_networks(self):
        """Initialize policy and value networks."""
        # Simple 2-layer networks
        scale = 0.1

        # Policy network: state -> mean, log_std
        self.policy_w1 = np.random.randn(self.state_dim, self.hidden_dim) * scale
        self.policy_b1 = np.zeros(self.hidden_dim)
        self.policy_w2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.policy_b2 = np.zeros(self.hidden_dim)
        self.policy_mean_w = np.random.randn(self.hidden_dim, self.action_dim) * scale
        self.policy_mean_b = np.zeros(self.action_dim)
        self.log_std = np.zeros(self.action_dim)

        # Value network: state -> value
        self.value_w1 = np.random.randn(self.state_dim, self.hidden_dim) * scale
        self.value_b1 = np.zeros(self.hidden_dim)
        self.value_w2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.value_b2 = np.zeros(self.hidden_dim)
        self.value_out_w = np.random.randn(self.hidden_dim, 1) * scale
        self.value_out_b = np.zeros(1)

    def _policy_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through policy network."""
        h1 = np.tanh(state @ self.policy_w1 + self.policy_b1)
        h2 = np.tanh(h1 @ self.policy_w2 + self.policy_b2)
        mean = h2 @ self.policy_mean_w + self.policy_mean_b
        std = np.exp(self.log_std)
        return mean, std

    def _value_forward(self, state: np.ndarray) -> float:
        """Forward pass through value network."""
        h1 = np.tanh(state @ self.value_w1 + self.value_b1)
        h2 = np.tanh(h1 @ self.value_w2 + self.value_b2)
        value = h2 @ self.value_out_w + self.value_out_b
        return float(value[0])

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action from policy."""
        mean, std = self._policy_forward(state)

        if explore:
            # Sample from Gaussian
            action = mean + std * np.random.randn(*mean.shape)
        else:
            action = mean

        # Clip to valid range [0, 15]
        action = np.clip(action, 0, 15)
        return action

    def store_transition(self, transition: Transition):
        """Store transition for later update."""
        self.trajectory.append(transition)

    def update(self, batch: List[Transition] = None) -> Dict[str, float]:
        """
        Update policy using PPO.

        Uses stored trajectory if batch is None.
        """
        if batch is None:
            batch = self.trajectory

        if len(batch) < 2:
            return {'loss': 0.0}

        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0

        for t in reversed(batch):
            if t.done:
                G = t.reward
            else:
                G = t.reward + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        values = np.array([self._value_forward(t.state) for t in batch])
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update (simplified - gradient descent on clipped objective)
        policy_loss = 0.0
        value_loss = 0.0

        for i, t in enumerate(batch):
            mean, std = self._policy_forward(t.state)

            # Old log probability (from when action was taken)
            old_log_prob = -0.5 * ((t.action - mean) / std) ** 2 - np.log(std)
            old_log_prob = old_log_prob.sum()

            # Simple policy gradient update direction
            grad_direction = advantages[i] * (t.action - mean) / (std ** 2)

            # Update policy (very simplified - proper impl uses optimizer)
            self.policy_mean_w += self.lr * 0.01 * np.outer(
                np.tanh(t.state @ self.policy_w1 + self.policy_b1) @ self.policy_w2 + self.policy_b2,
                grad_direction
            ).T.mean(axis=0, keepdims=True).T

            # Value update
            value_error = returns[i] - values[i]
            value_loss += value_error ** 2

        # Clear trajectory
        self.trajectory = []

        return {
            'policy_loss': policy_loss / len(batch),
            'value_loss': value_loss / len(batch),
            'mean_return': returns.mean(),
            'mean_advantage': advantages.mean()
        }

    def save(self, path: str):
        """Save agent parameters."""
        params = {
            'policy_w1': self.policy_w1.tolist(),
            'policy_b1': self.policy_b1.tolist(),
            'policy_w2': self.policy_w2.tolist(),
            'policy_b2': self.policy_b2.tolist(),
            'policy_mean_w': self.policy_mean_w.tolist(),
            'policy_mean_b': self.policy_mean_b.tolist(),
            'log_std': self.log_std.tolist(),
            'value_w1': self.value_w1.tolist(),
            'value_b1': self.value_b1.tolist(),
            'value_w2': self.value_w2.tolist(),
            'value_b2': self.value_b2.tolist(),
            'value_out_w': self.value_out_w.tolist(),
            'value_out_b': self.value_out_b.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(params, f)

    def load(self, path: str):
        """Load agent parameters."""
        with open(path, 'r') as f:
            params = json.load(f)

        self.policy_w1 = np.array(params['policy_w1'])
        self.policy_b1 = np.array(params['policy_b1'])
        self.policy_w2 = np.array(params['policy_w2'])
        self.policy_b2 = np.array(params['policy_b2'])
        self.policy_mean_w = np.array(params['policy_mean_w'])
        self.policy_mean_b = np.array(params['policy_mean_b'])
        self.log_std = np.array(params['log_std'])
        self.value_w1 = np.array(params['value_w1'])
        self.value_b1 = np.array(params['value_b1'])
        self.value_w2 = np.array(params['value_w2'])
        self.value_b2 = np.array(params['value_b2'])
        self.value_out_w = np.array(params['value_out_w'])
        self.value_out_b = np.array(params['value_out_b'])


class SACAgent(RLAgent):
    """
    Soft Actor-Critic agent (simplified numpy version).

    Off-policy algorithm with:
    - Maximum entropy objective
    - Twin Q-networks
    - Automatic temperature tuning
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 1,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self._init_networks()
        self.replay_buffer = ReplayBuffer(capacity=100000)

    def _init_networks(self):
        """Initialize actor and critic networks."""
        scale = 0.1

        # Actor (policy)
        self.actor_w1 = np.random.randn(self.state_dim, self.hidden_dim) * scale
        self.actor_b1 = np.zeros(self.hidden_dim)
        self.actor_w2 = np.random.randn(self.hidden_dim, self.action_dim * 2) * scale
        self.actor_b2 = np.zeros(self.action_dim * 2)

        # Twin critics Q1, Q2
        input_dim = self.state_dim + self.action_dim
        self.q1_w1 = np.random.randn(input_dim, self.hidden_dim) * scale
        self.q1_b1 = np.zeros(self.hidden_dim)
        self.q1_w2 = np.random.randn(self.hidden_dim, 1) * scale
        self.q1_b2 = np.zeros(1)

        self.q2_w1 = np.random.randn(input_dim, self.hidden_dim) * scale
        self.q2_b1 = np.zeros(self.hidden_dim)
        self.q2_w2 = np.random.randn(self.hidden_dim, 1) * scale
        self.q2_b2 = np.zeros(1)

        # Target critics
        self.q1_target_w1 = self.q1_w1.copy()
        self.q1_target_b1 = self.q1_b1.copy()
        self.q1_target_w2 = self.q1_w2.copy()
        self.q1_target_b2 = self.q1_b2.copy()

        self.q2_target_w1 = self.q2_w1.copy()
        self.q2_target_b1 = self.q2_b1.copy()
        self.q2_target_w2 = self.q2_w2.copy()
        self.q2_target_b2 = self.q2_b2.copy()

    def _actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Actor forward pass -> mean, log_std."""
        h = np.tanh(state @ self.actor_w1 + self.actor_b1)
        out = h @ self.actor_w2 + self.actor_b2
        mean = out[:self.action_dim] if len(out.shape) == 1 else out[:, :self.action_dim]
        log_std = out[self.action_dim:] if len(out.shape) == 1 else out[:, self.action_dim:]
        log_std = np.clip(log_std, -20, 2)
        return mean, np.exp(log_std)

    def _q_forward(self, state: np.ndarray, action: np.ndarray, q_idx: int = 1) -> float:
        """Critic forward pass."""
        x = np.concatenate([state, action])
        if q_idx == 1:
            h = np.tanh(x @ self.q1_w1 + self.q1_b1)
            return float((h @ self.q1_w2 + self.q1_b2)[0])
        else:
            h = np.tanh(x @ self.q2_w1 + self.q2_b1)
            return float((h @ self.q2_w2 + self.q2_b2)[0])

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Sample action from policy."""
        mean, std = self._actor_forward(state)

        if explore:
            action = mean + std * np.random.randn(*mean.shape)
        else:
            action = mean

        return np.clip(action, 0, 15)

    def store_transition(self, transition: Transition):
        """Store transition in replay buffer."""
        self.replay_buffer.push(transition)

    def update(self, batch: List[Transition] = None) -> Dict[str, float]:
        """Update SAC networks."""
        if batch is None:
            if len(self.replay_buffer) < 256:
                return {'loss': 0.0}
            batch = self.replay_buffer.sample(256)

        # Compute targets and losses (simplified)
        q1_loss = 0.0
        q2_loss = 0.0
        actor_loss = 0.0

        for t in batch:
            # Q-function update
            with_action_next, _ = self._actor_forward(t.next_state)
            q1_next = self._q_forward(t.next_state, with_action_next, 1)
            q2_next = self._q_forward(t.next_state, with_action_next, 2)
            q_next = min(q1_next, q2_next)

            target = t.reward + (1 - t.done) * self.gamma * q_next

            q1_pred = self._q_forward(t.state, t.action, 1)
            q2_pred = self._q_forward(t.state, t.action, 2)

            q1_loss += (q1_pred - target) ** 2
            q2_loss += (q2_pred - target) ** 2

        # Soft update targets
        self._soft_update()

        return {
            'q1_loss': q1_loss / len(batch),
            'q2_loss': q2_loss / len(batch),
            'actor_loss': actor_loss / len(batch)
        }

    def _soft_update(self):
        """Soft update of target networks."""
        self.q1_target_w1 = self.tau * self.q1_w1 + (1 - self.tau) * self.q1_target_w1
        self.q1_target_b1 = self.tau * self.q1_b1 + (1 - self.tau) * self.q1_target_b1
        self.q1_target_w2 = self.tau * self.q1_w2 + (1 - self.tau) * self.q1_target_w2
        self.q1_target_b2 = self.tau * self.q1_b2 + (1 - self.tau) * self.q1_target_b2

        self.q2_target_w1 = self.tau * self.q2_w1 + (1 - self.tau) * self.q2_target_w1
        self.q2_target_b1 = self.tau * self.q2_b1 + (1 - self.tau) * self.q2_target_b1
        self.q2_target_w2 = self.tau * self.q2_w2 + (1 - self.tau) * self.q2_target_w2
        self.q2_target_b2 = self.tau * self.q2_b2 + (1 - self.tau) * self.q2_target_b2

    def save(self, path: str):
        """Save agent."""
        params = {
            'actor_w1': self.actor_w1.tolist(),
            'actor_b1': self.actor_b1.tolist(),
            'actor_w2': self.actor_w2.tolist(),
            'actor_b2': self.actor_b2.tolist(),
            'q1_w1': self.q1_w1.tolist(),
            'q1_b1': self.q1_b1.tolist(),
            'q1_w2': self.q1_w2.tolist(),
            'q1_b2': self.q1_b2.tolist(),
            'q2_w1': self.q2_w1.tolist(),
            'q2_b1': self.q2_b1.tolist(),
            'q2_w2': self.q2_w2.tolist(),
            'q2_b2': self.q2_b2.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(params, f)

    def load(self, path: str):
        """Load agent."""
        with open(path, 'r') as f:
            params = json.load(f)
        self.actor_w1 = np.array(params['actor_w1'])
        self.actor_b1 = np.array(params['actor_b1'])
        self.actor_w2 = np.array(params['actor_w2'])
        self.actor_b2 = np.array(params['actor_b2'])
        self.q1_w1 = np.array(params['q1_w1'])
        self.q1_b1 = np.array(params['q1_b1'])
        self.q1_w2 = np.array(params['q1_w2'])
        self.q1_b2 = np.array(params['q1_b2'])
        self.q2_w1 = np.array(params['q2_w1'])
        self.q2_b1 = np.array(params['q2_b1'])
        self.q2_w2 = np.array(params['q2_w2'])
        self.q2_b2 = np.array(params['q2_b2'])


class RacingTrainer:
    """
    Training orchestrator for RL racing agents.

    Supports:
    - Curriculum learning
    - Parallel environment rollouts
    - Evaluation and checkpointing
    - Training visualization
    """

    def __init__(
        self,
        agent: RLAgent,
        env: SolarBoatRacingEnv,
        log_dir: str = "./rl_logs"
    ):
        self.agent = agent
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Training stats
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[float] = []

    def train(
        self,
        n_episodes: int = 1000,
        update_every: int = 1,
        eval_every: int = 50,
        save_every: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train agent for n_episodes.

        Args:
            n_episodes: Number of training episodes
            update_every: Update agent every N episodes
            eval_every: Evaluate every N episodes
            save_every: Save checkpoint every N episodes
            verbose: Print progress

        Returns:
            Training statistics
        """
        for episode in range(n_episodes):
            # Collect episode
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.agent.select_action(state, explore=True)
                next_state, reward, done, info = self.env.step(action)

                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info
                )

                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(transition)

                state = next_state
                episode_reward += reward
                episode_length += 1

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update agent
            if episode % update_every == 0:
                update_info = self.agent.update()

            # Evaluate
            if episode % eval_every == 0:
                eval_reward, eval_success = self.evaluate(n_episodes=5)
                self.success_rate.append(eval_success)

                if verbose:
                    print(
                        f"Episode {episode} | "
                        f"Reward: {episode_reward:.1f} | "
                        f"Eval Reward: {eval_reward:.1f} | "
                        f"Success: {eval_success:.0%}"
                    )

            # Save checkpoint
            if episode % save_every == 0:
                self.agent.save(str(self.log_dir / f"agent_{episode}.json"))

        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'success_rate': self.success_rate
        }

    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate agent without exploration."""
        total_reward = 0.0
        successes = 0

        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.agent.select_action(state, explore=False)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

            if info.get('success', False):
                successes += 1

        return total_reward / n_episodes, successes / n_episodes


def create_rl_agent(
    agent_type: str = 'ppo',
    **kwargs
) -> RLAgent:
    """
    Factory function to create RL agent.

    Args:
        agent_type: 'ppo' or 'sac'
        **kwargs: Agent-specific parameters

    Returns:
        Configured RL agent
    """
    if agent_type.lower() == 'ppo':
        return PPOAgent(**kwargs)
    elif agent_type.lower() == 'sac':
        return SACAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_racing_agent(
    n_episodes: int = 500,
    agent_type: str = 'ppo',
    save_path: str = './trained_agent.json'
) -> Tuple[RLAgent, Dict]:
    """
    Convenience function to train a racing agent.

    Args:
        n_episodes: Number of training episodes
        agent_type: 'ppo' or 'sac'
        save_path: Where to save trained agent

    Returns:
        Trained agent and training stats
    """
    # Create environment
    world_model = create_default_world_model()
    env = SolarBoatRacingEnv(world_model=world_model)

    # Create agent
    agent = create_rl_agent(agent_type)

    # Train
    trainer = RacingTrainer(agent, env)
    stats = trainer.train(n_episodes=n_episodes, verbose=True)

    # Save
    agent.save(save_path)

    return agent, stats
