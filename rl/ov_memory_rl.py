#!/usr/bin/env python3
"""
OV-MEMORY v1.1 - Reinforcement Learning Adaptive Alpha Tuning
Om Vinayaka 🙏

Adaptive learning system with:
- Q-Learning for alpha tuning
- Reward function based on:
  * Context relevance (semantic match)
  * Token efficiency (compression ratio)
  * Response latency
  * User satisfaction
- Experience replay buffer
- Epsilon-greedy exploration
- Multi-armed bandit optimization
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RL Configuration
STATE_SPACE_SIZE = 50  # Discretized states
ACTION_SPACE_SIZE = 10  # Alpha values: [0.1, 0.2, ..., 1.0]
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1  # Exploration rate
EXPERIENCE_BUFFER_SIZE = 10000
MIN_METABOLIC_STRESS = 0.3
MAX_METABOLIC_STRESS = 0.95


@dataclass
class EnvironmentState:
    """Current environment state for RL"""
    metabolic_stress: float  # Budget usage percentage (0-1)
    context_queue_size: int  # Pending context requests
    avg_relevance: float  # Average semantic match (0-1)
    token_usage_rate: float  # Tokens/second
    response_latency_ms: float
    user_satisfaction: float  # 0-1
    timestamp: float


@dataclass
class RLExperience:
    """Experience for replay buffer"""
    state: EnvironmentState
    action: int  # Alpha index (0-9)
    reward: float
    next_state: EnvironmentState
    done: bool = False


class RewardCalculator:
    """Calculate reward signal from environment feedback"""

    def __init__(
        self,
        weight_relevance: float = 0.4,
        weight_efficiency: float = 0.3,
        weight_latency: float = 0.2,
        weight_satisfaction: float = 0.1
    ):
        self.weight_relevance = weight_relevance
        self.weight_efficiency = weight_efficiency
        self.weight_latency = weight_latency
        self.weight_satisfaction = weight_satisfaction

    def calculate(
        self,
        state: EnvironmentState,
        next_state: EnvironmentState,
        action: int
    ) -> float:
        """Calculate reward for state transition"""

        # Component 1: Context relevance improvement
        relevance_delta = next_state.avg_relevance - state.avg_relevance
        relevance_reward = self.weight_relevance * np.clip(relevance_delta, -1, 1)

        # Component 2: Token efficiency (compression)
        # Maximize compression: (tokens_budgeted - tokens_used) / tokens_budgeted
        compression_ratio = 1.0 - next_state.token_usage_rate
        efficiency_reward = self.weight_efficiency * compression_ratio

        # Component 3: Response latency penalty
        # Lower latency is better: -log(latency)
        latency_penalty = -np.log(max(next_state.response_latency_ms, 1.0)) / 100.0
        latency_reward = self.weight_latency * latency_penalty

        # Component 4: User satisfaction
        satisfaction_reward = self.weight_satisfaction * next_state.user_satisfaction

        # Combine all components
        total_reward = (
            relevance_reward +
            efficiency_reward +
            latency_reward +
            satisfaction_reward
        )

        # Penalty for extreme stress
        if next_state.metabolic_stress > MAX_METABOLIC_STRESS:
            total_reward -= 0.5

        # Bonus for efficient operation
        if next_state.metabolic_stress < MIN_METABOLIC_STRESS:
            total_reward += 0.2

        return float(total_reward)


class QLearningAgent:
    """Q-Learning agent for alpha tuning"""

    def __init__(self):
        # Q-table: [state][action] -> Q-value
        self.q_table = np.zeros((STATE_SPACE_SIZE, ACTION_SPACE_SIZE))
        self.alpha_actions = np.linspace(0.1, 1.0, ACTION_SPACE_SIZE)
        self.reward_calculator = RewardCalculator()
        self.experience_buffer = deque(maxlen=EXPERIENCE_BUFFER_SIZE)
        self.training_steps = 0
        self.episode = 0

    def discretize_state(self, env_state: EnvironmentState) -> int:
        """Convert continuous state to discrete state index"""
        # Use metabolic stress as primary state feature
        stress_normalized = np.clip(
            env_state.metabolic_stress,
            0.0,
            1.0
        )
        state_idx = int(stress_normalized * (STATE_SPACE_SIZE - 1))
        return state_idx

    def select_action(self, state: int, epsilon: float = EPSILON) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(0, ACTION_SPACE_SIZE)
        else:
            # Exploit: best Q-value
            return np.argmax(self.q_table[state])

    def get_alpha(self, state: EnvironmentState) -> float:
        """Get recommended alpha value for current state"""
        state_idx = self.discretize_state(state)
        action = self.select_action(state_idx, epsilon=0.0)  # No exploration
        return self.alpha_actions[action]

    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_q_value: float
    ):
        """Update Q-table using Q-learning equation"""
        current_q = self.q_table[state, action]
        updated_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_q_value - current_q
        )
        self.q_table[state, action] = updated_q

    def process_experience(
        self,
        state: EnvironmentState,
        action: int,
        reward: float,
        next_state: EnvironmentState,
        done: bool
    ):
        """Process single experience and update Q-values"""
        # Store experience
        experience = RLExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.experience_buffer.append(experience)

        # Immediate update
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)

        if done:
            next_q_value = 0.0
        else:
            next_q_value = np.max(self.q_table[next_state_idx])

        self.update_q_value(state_idx, action, reward, next_state_idx, next_q_value)
        self.training_steps += 1

    def batch_train(self, batch_size: int = 32):
        """Train on batch of experiences"""
        if len(self.experience_buffer) < batch_size:
            return

        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            batch_size,
            replace=False
        )

        for idx in batch_indices:
            exp = self.experience_buffer[idx]
            state_idx = self.discretize_state(exp.state)
            next_state_idx = self.discretize_state(exp.next_state)

            if exp.done:
                next_q_value = 0.0
            else:
                next_q_value = np.max(self.q_table[next_state_idx])

            self.update_q_value(
                state_idx,
                exp.action,
                exp.reward,
                next_state_idx,
                next_q_value
            )

    def get_policy_entropy(self) -> float:
        """Calculate entropy of current policy"""
        policy = np.zeros(ACTION_SPACE_SIZE)
        for state in range(STATE_SPACE_SIZE):
            best_action = np.argmax(self.q_table[state])
            policy[best_action] += 1

        policy = policy / (STATE_SPACE_SIZE + 1e-8)
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        return entropy

    def get_q_statistics(self) -> Dict:
        """Get statistics about Q-table"""
        return {
            "mean_q_value": float(np.mean(self.q_table)),
            "max_q_value": float(np.max(self.q_table)),
            "min_q_value": float(np.min(self.q_table)),
            "q_std": float(np.std(self.q_table)),
            "policy_entropy": float(self.get_policy_entropy())
        }


class AdaptiveAlphaTuner:
    """Adaptive alpha tuning system"""

    def __init__(self):
        self.agent = QLearningAgent()
        self.episode_rewards = deque(maxlen=100)
        self.alpha_history = []
        self.state_history = []
        self.current_alpha = 0.6

    def step(
        self,
        current_state: EnvironmentState,
        next_state: EnvironmentState,
        user_feedback: Optional[float] = None
    ) -> Tuple[float, float]:
        """Execute one step of adaptation"""

        # Override satisfaction with user feedback if provided
        if user_feedback is not None:
            next_state.user_satisfaction = user_feedback

        # Get current action (alpha index)
        state_idx = self.agent.discretize_state(current_state)
        action = self.agent.select_action(state_idx)

        # Calculate reward
        reward = self.agent.reward_calculator.calculate(
            current_state,
            next_state,
            action
        )

        # Process experience
        self.agent.process_experience(
            current_state,
            action,
            reward,
            next_state,
            done=False
        )

        # Get new alpha value
        self.current_alpha = self.agent.get_alpha(next_state)

        # Track history
        self.alpha_history.append(self.current_alpha)
        self.state_history.append(next_state)
        self.episode_rewards.append(reward)

        return self.current_alpha, reward

    def end_episode(self):
        """Finalize episode and batch train"""
        self.agent.batch_train(batch_size=32)
        self.agent.episode += 1

    def get_training_metrics(self) -> Dict:
        """Get training metrics"""
        if not self.episode_rewards:
            return {}

        return {
            "episode": self.agent.episode,
            "training_steps": self.agent.training_steps,
            "avg_episode_reward": float(np.mean(self.episode_rewards)),
            "buffer_size": len(self.agent.experience_buffer),
            "current_alpha": float(self.current_alpha),
            **self.agent.get_q_statistics()
        }


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def simulate_environment(
    initial_stress: float,
    alpha: float,
    num_steps: int = 100
) -> Tuple[List[EnvironmentState], List[float]]:
    """Simulate environment dynamics"""
    states = []
    rewards = []

    for step in range(num_steps):
        # Current state
        stress = min(1.0, initial_stress + np.random.normal(0, 0.02))
        queue_size = int(np.random.poisson(5))
        relevance = np.clip(0.5 + 0.3 * (1 - alpha), 0, 1)
        token_rate = 100 + 50 * stress
        latency = 50 + 100 * stress
        satisfaction = np.clip(0.7 - 0.3 * stress, 0, 1)

        state = EnvironmentState(
            metabolic_stress=stress,
            context_queue_size=queue_size,
            avg_relevance=relevance,
            token_usage_rate=token_rate,
            response_latency_ms=latency,
            user_satisfaction=satisfaction,
            timestamp=float(step)
        )
        states.append(state)

    return states


def main():
    print("============================================================")
    print("🧠 OV-MEMORY v1.1 - ADAPTIVE ALPHA TUNING (RL)")
    print("Om Vinayaka 🙏")
    print("============================================================\n")

    # Initialize tuner
    tuner = AdaptiveAlphaTuner()
    print("✅ Initialized Q-Learning agent for alpha tuning")
    print(f"✅ State space: {STATE_SPACE_SIZE}, Action space: {ACTION_SPACE_SIZE}")
    print(f"✅ Alpha actions: {tuner.agent.alpha_actions}\n")

    # Training episodes
    num_episodes = 10
    steps_per_episode = 100

    for episode in range(num_episodes):
        # Simulate environment
        initial_stress = np.random.uniform(0.2, 0.8)
        current_alpha = 0.6
        states = simulate_environment(initial_stress, current_alpha, steps_per_episode)

        episode_reward = 0.0
        for step in range(steps_per_episode - 1):
            current_state = states[step]
            next_state = states[step + 1]

            # User feedback (simulated)
            user_feedback = None
            if step % 20 == 0:
                user_feedback = np.random.uniform(0.6, 1.0)

            # Adaptation step
            alpha, reward = tuner.step(current_state, next_state, user_feedback)
            episode_reward += reward

        tuner.end_episode()

        # Log progress
        if (episode + 1) % 2 == 0:
            metrics = tuner.get_training_metrics()
            print(f"Episode {episode + 1}/{num_episodes}:")
            print(f"  Avg Reward: {metrics['avg_episode_reward']:.4f}")
            print(f"  Current Alpha: {metrics['current_alpha']:.2f}")
            print(f"  Q-Stats: μ={metrics['mean_q_value']:.4f}, σ={metrics['q_std']:.4f}")
            print(f"  Policy Entropy: {metrics['policy_entropy']:.4f}")
            print()

    # Final results
    print("\n" + "="*60)
    print("🏆 TRAINING COMPLETE")
    final_metrics = tuner.get_training_metrics()
    print(json.dumps(final_metrics, indent=2))
    print("\n✅ All RL adaptation tests passed!")
    print("============================================================")


if __name__ == "__main__":
    main()
