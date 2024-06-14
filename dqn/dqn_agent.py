import logging
import os
import torch
import torch.optim as optim
import numpy as np
import random
import pickle

from torch import nn

from exceptions.model_memory_load_error import ModelMemoryLoadError
from dqn.dqn import DQN
from dqn.replay_memory import ReplayMemory


class DQNAgent:
    """
    DQN Agent for training and making decisions in the game environment.
    """

    def __init__(self, state_size, action_size, update_target_frequency=10):
        """
        Initialize the DQN Agent with specified state and action sizes.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            update_target_frequency (int): Frequency of updating the target model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_frequency = update_target_frequency
        self.step_counter = 0

    def update_target_model(self):
        """
        Update the target model with the current model's weights.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory and update the target model periodically.

        Args:
            state (list): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (list): The next state.
            done (bool): Whether the episode is finished.
        """
        self.memory.add(state, action, reward, next_state, done)
        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()

    def act(self, state):
        """
        Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (list): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        """
        Train the model on a batch of experiences from the replay memory.

        Args:
            batch_size (int): The number of experiences to sample.
        """
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f'Episode: {self.step_counter // batch_size}, Epsilon: {self.epsilon}')

    def save(self, model_path, memory_path):
        """
        Save the current model and replay memory to disk.

        Args:
            model_path (str): Path to save the model.
            memory_path (str): Path to save the replay memory.
        """
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)
        print("Model and memory saved.")

    def load(self, model_path, memory_path):
        """
        Load the model and replay memory from disk.

        Args:
            model_path (str): Path to load the model from.
            memory_path (str): Path to load the replay memory from.
        """
        if os.path.exists(model_path) and os.path.exists(memory_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                with open(memory_path, 'rb') as memory_file:
                    self.memory = pickle.load(memory_file)
                print("Model and memory loaded.")
            except ModelMemoryLoadError as e:
                print(f"Error loading model/memory: {e}. Starting from scratch.")
        else:
            print("No saved model/memory found, starting from scratch.")
