import gymnasium as gym
import numpy as np
from snake_env import SnakeEnv
import time

class SimpleQLearningAgent:
    """A simple Q-learning agent for the snake environment."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_size = action_size
    
    def get_state_key(self, observation):
        """Convert observation to a hashable state key."""
        # Create a simple state representation
        head = None
        food = None
        
        # Find snake head and food
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if observation[i, j] == 1:  # Snake head (first occurrence)
                    if head is None:
                        head = (i, j)
                elif observation[i, j] == 2:  # Food
                    food = (i, j)
        
        if head is None or food is None:
            return "unknown"
        
        # Create relative direction to food
        dx = food[1] - head[1]
        dy = food[0] - head[0]
        
        # Discretize direction
        if abs(dx) > abs(dy):
            direction = "horizontal"
        else:
            direction = "vertical"
        
        return f"{direction}_{dx}_{dy}"
    
    def get_action(self, observation):
        """Choose action using epsilon-greedy policy."""
        state_key = self.get_state_key(observation)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, observation, action, reward, next_observation):
        """Update Q-values."""
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

def train_agent(episodes=1000, render=False):
    """Train a Q-learning agent on the snake environment."""
    env = SnakeEnv(render_mode="human" if render else None)
    agent = SimpleQLearningAgent(
        state_size=env.observation_space.shape[0] * env.observation_space.shape[1],
        action_size=env.action_space.n,
        learning_rate=0.1,
        epsilon=0.1
    )
    
    scores = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            
            agent.update(observation, action, reward, next_observation)
            
            observation = next_observation
            total_reward += reward
            
            if render:
                env.render()
                time.sleep(0.05)
        
        scores.append(info['score'])
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score (last 100): {avg_score:.2f}")
    
    env.close()
    return agent, scores

def test_trained_agent(agent, episodes=5):
    """Test the trained agent."""
    env = SnakeEnv(render_mode="human")
    
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"Testing episode {episode + 1}")
        
        while not done:
            action = agent.get_action(observation)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            env.render()
            time.sleep(0.1)
        
        print(f"Episode {episode + 1} finished with score: {info['score']}")
    
    env.close()

if __name__ == "__main__":
    print("Training Q-learning agent on Snake environment...")
    
    # Train without rendering for speed
    agent, scores = train_agent(episodes=500, render=False)
    
    print("\nTraining completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_trained_agent(agent, episodes=3) 