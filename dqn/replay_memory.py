from collections import deque
import numpy as np


class ReplayMemory:
    """
    Replay memory for storing experiences for training the DQN agent.
    """

    def __init__(self, capacity):
        """
        Initialize the replay memory with a specified capacity.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the replay memory.

        Args:
            state (list): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (list): The next state.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: Batches of states, actions, rewards, next states, and done flags.
        """
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        """
        Get the current size of the replay memory.

        Returns:
            int: The number of experiences stored.
        """
        return len(self.memory)
