### DQNAgent Class

```python
class DQNAgent:
    def __init__(self, state_size, action_size, update_target_frequency=10):
```
- **__init__**: Constructor initializes the agent.
  - `state_size`: Dimension of the state space.
  - `action_size`: Number of possible actions.
  - `update_target_frequency`: Frequency of updating the target network.

```python
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- **device**: Determines if a GPU (CUDA) is available for computation.

```python
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
```
- **state_size**: Stores the size of the state space.
- **action_size**: Stores the number of actions.
- **memory**: Initializes replay memory with a capacity of 10,000 experiences.
- **gamma**: Discount factor for future rewards.
- **epsilon**: Initial exploration rate.
- **epsilon_min**: Minimum exploration rate.
- **epsilon_decay**: Decay rate for epsilon.
- **learning_rate**: Learning rate for the optimizer.

```python
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
```
- **model**: The main Q-network.
- **target_model**: The target Q-network.
- **update_target_model()**: Copies weights from `model` to `target_model`.
- **optimizer**: Adam optimizer for training the network.

```python
        self.update_target_frequency = update_target_frequency
        self.step_counter = 0
```
- **update_target_frequency**: Frequency of updating the target model.
- **step_counter**: Counts the number of steps taken.

```python
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
```
- **update_target_model**: Copies the weights from the main model to the target model.

```python
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()
```
- **remember**: Stores experience in replay memory and updates the target model periodically.

```python
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
```
- **act**: Chooses an action using an epsilon-greedy policy. It either explores (random action) or exploits (best action according to the model).

```python
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
```
- **replay**: Trains the model using experiences sampled from replay memory.
  - Checks if there are enough samples in memory.
  - Converts experiences to PyTorch tensors and moves them to the appropriate device.

```python
        q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
```
- **q_values**: Q-values of the current states for the selected actions.
- **max_next_q_values**: Maximum Q-values of the next states from the target model.
- **target_q_values**: Target Q-values using the Bellman equation.

```python
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f'Episode: {self.step_counter // batch_size}, Epsilon: {self.epsilon}')
```
- **loss**: Mean Squared Error (MSE) loss between predicted and target Q-values.
- **optimizer.zero_grad()**: Clears previous gradients.
- **loss.backward()**: Backpropagates the loss.
- **optimizer.step()**: Updates the model weights.
- **epsilon decay**: Reduces exploration rate over time.
- **logging**: Logs the current episode and epsilon value.

```python
    def save(self, model_path, memory_path):
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)
        print("Model and memory saved.")
```
- **save**: Saves the model weights and replay memory to files.

```python
    def load(self, model_path, memory_path):
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
```
- **load**: Loads the model weights and replay memory from files if they exist.

### GameEnvironment Class

```python
class GameEnvironment:
    def __init__(self, game):
        self.game = game
        self.previous_distance = None
        self.previous_player_deaths = 0
```
- **__init__**: Initializes the game environment.
  - `game`: The game instance.
  - `previous_distance`: Tracks the player's distance to the target.
  - `previous_player_deaths`: Tracks the number of player deaths.

```python
    def reset(self):
        self.game.reset()
        self.previous_distance = self.calculate_distance_to_target(self.game.player)
        return self.get_state()
```
- **reset**: Resets the game and returns the initial state.

```python
    def get_state(self):
        player = self.game.player
        state = [player.x, player.y, player.dx, player.dy]
        for enemy in self.game.levels[self.game.current_level_index].enemies:
            state.extend([enemy.x, enemy.y, enemy.dx, enemy.dy])
        return state
```
- **get_state**: Constructs the state representation from the player's and enemies' positions and velocities.

```python
    def step(self, action):
        keys = {K_LEFT: False, K_RIGHT: False, K_UP: False, K_DOWN: False}
        if action == 0:
            keys[K_LEFT] = True
        elif action == 1:
            keys[K_RIGHT] = True
        elif action == 2:
            keys[K_UP] = True
        elif action == 3:
            keys[K_DOWN] = True

        self.game.player.dx, self.game.player.dy = 0, 0
        if keys[K_LEFT]:
            self.game.player.dx = -2
        if keys[K_RIGHT]:
            self.game.player.dx = 2
        if keys[K_UP]:
            self.game.player.dy = -2
        if keys[K_DOWN]:
            self.game.player.dy = 2

        self.game.update()
        next_state = self.get_state()
```
- **step**: Takes an action and updates the game state.
  - Sets the player's velocity based on the action.
  - Updates the game.
  - Retrieves the next state.

```python
        reward, done = self.calculate_reward()
        if done or self.previous_player_deaths - self.game.player.player_deaths > 0:
            self.reset()  # Reset the game on death

        return next_state, reward, done
```
- **reward**: Calculates the reward and checks if the episode is done.
- **reset**: Resets the game if the player dies.
- **return**: Returns the next state, reward, and done flag.

```python
    def calculate_distance_to_target(self, player):
        target = self.game.levels[self.game.current_level_index].target_zone
        return math.sqrt((player.x - target.x) ** 2 + (player.y - target.y) ** 2)
```
- **calculate_distance_to_target**: Computes the Euclidean distance from the player to the target zone.

```python
    def calculate_reward(self):
        player = self.game.player
        current_distance = self.calculate_distance_to_target(player)

        reward = -0.1  # Default small penalty to

 encourage movement
        done = False

        if player.isFinished:
            reward += 200
            done = True
        elif player.player_deaths - self.previous_player_deaths > 0:
            reward -= 100
            self.previous_player_deaths = player.player_deaths
        if self.previous_distance is not None:
            distance_reward = self.previous_distance - current_distance
            reward += distance_reward * 10
        self.previous_distance = current_distance

        return reward, done
```
- **calculate_reward**: Calculates the reward based on the player's status and movement.
  - Penalizes by default to encourage movement.
  - Rewards for reaching the target.
  - Penalizes for dying.
  - Rewards for reducing distance to the target.

```python
    def render(self):
        self.game.game_observer.update()
```
- **render**: Updates the game display.

### ReplayMemory Class

```python
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
```
- **__init__**: Initializes the replay memory.
  - `capacity`: Maximum size of the memory.

```python
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
```
- **add**: Adds an experience to the memory.

```python
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
```
- **sample**: Samples a batch of experiences from the memory.

```python
    def __len__(self):
        return len(self.memory)
```
- **__len__**: Returns the current size of the memory.

### train_dqn Function

```python
def train_dqn(game, episodes, batch_size, max_steps_per_episode=500):
    state_size = 4 + 4 * len(game.levels[game.current_level_index].enemies)
    action_size = 4  # [left, right, up, down]
    agent = DQNAgent(state_size, action_size)
    env = GameEnvironment(game)
    model_path = "dqn_model.pth"
    memory_path = "replay_memory.pkl"
```
- **train_dqn**: Function to train the DQN agent.
  - `state_size`: State size based on player and enemies.
  - `action_size`: Number of possible actions.
  - `agent`: Initializes the DQN agent.
  - `env`: Initializes the game environment.
  - `model_path` and `memory_path`: File paths to save the model and memory.

```python
    # agent.load(model_path, memory_path) # Uncomment this line to load the model and memory

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time_step in range(max_steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            env.render()

        tracker.save_episode_data(e, total_reward, agent.epsilon, game.finish_count)
        print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
        agent.replay(batch_size)
```
- **Training loop**:
  - Resets the environment.
  - Iterates through time steps, getting user events (quitting the game), selecting an action, taking a step in the environment, storing experience, rendering the game, and updating the agent.
  - Saves episode data and prints progress.
  - Trains the agent using the replay memory.

```python
    agent.save(model_path, memory_path)
```
- **save**: Saves the trained model and memory.

### Main Function

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple 2D Game")
    parser.add_argument('--mode', choices=['play', 'train', 'plot'], required=True, help="Mode to run the game in")
    args = parser.parse_args()

    try:
        main(args.mode)
    except KeyboardInterrupt:
        print("Game terminated by user")
```
- **Main function**: Parses command-line arguments to run the game in different modes and handles termination by the user.

### Summary

The provided code implements a DQN agent that interacts with a game environment for training and gameplay. The agent uses a replay memory to store experiences and trains the Q-network using batches of these experiences. The game environment provides state information and calculates rewards based on the player's actions and status. The training function orchestrates the interaction between the agent and the environment, saving and loading models as necessary.


## DQNAgent In-depth

#### `__init__` Method

```python
class DQNAgent:
    def __init__(self, state_size, action_size, update_target_frequency=10):
```
- **__init__**: Constructor for initializing the DQN agent.
  - `state_size`: The dimensionality of the state space.
  - `action_size`: The number of possible actions.
  - `update_target_frequency`: Frequency (in steps) for updating the target network.

```python
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- **self.device**: Determines whether to use a GPU (if available) or CPU for computations.

```python
        self.state_size = state_size
        self.action_size = action_size
```
- **self.state_size**: Stores the state size.
- **self.action_size**: Stores the action size.

```python
        self.memory = ReplayMemory(10000)
```
- **self.memory**: Initializes the replay memory with a capacity of 10,000 experiences. This memory is used to store experiences and sample mini-batches for training.

```python
        self.gamma = 0.99
```
- **self.gamma**: Discount factor for future rewards, which determines the importance of future rewards.

```python
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
```
- **self.epsilon**: Initial exploration rate for the epsilon-greedy policy.
- **self.epsilon_min**: Minimum value for epsilon, ensuring some level of exploration.
- **self.epsilon_decay**: Decay rate for epsilon to reduce exploration over time.

```python
        self.learning_rate = 0.001
```
- **self.learning_rate**: Learning rate for the optimizer, controlling how much to update the network's weights during training.

```python
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
```
- **self.model**: The primary Q-network for predicting Q-values.
- **self.target_model**: The target Q-network used to provide stable target values during training.
- **self.update_target_model()**: Copies weights from the primary Q-network to the target Q-network.

```python
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
```
- **self.optimizer**: Adam optimizer for training the Q-network with the specified learning rate.

```python
        self.update_target_frequency = update_target_frequency
```
- **self.update_target_frequency**: Stores the frequency at which the target model should be updated.

```python
        self.step_counter = 0
```
- **self.step_counter**: Keeps track of the number of steps taken. This is used to determine when to update the target model.

#### `update_target_model` Method

```python
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
```
- **update_target_model**: Copies the weights from the primary Q-network to the target Q-network. This helps stabilize the training process by providing fixed targets for a certain number of steps.

#### `remember` Method

```python
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()
```
- **remember**: Stores the experience in replay memory and increments the step counter.
  - `state`, `action`, `reward`, `next_state`, `done`: Components of the experience.
  - Adds the experience to replay memory.
  - Checks if the target model needs to be updated based on the step counter and update frequency.

#### `act` Method

```python
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
```
- **act**: Chooses an action using an epsilon-greedy policy.
  - With probability `epsilon`, selects a random action to encourage exploration.
  - Otherwise, uses the Q-network to predict the action with the highest Q-value.
  - `state` is converted to a PyTorch tensor and moved to the appropriate device.
  - `act_values` contains the Q-values predicted by the model for the given state.
  - Returns the action with the highest predicted Q-value.

#### `replay` Method

```python
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
```
- **replay**: Trains the Q-network using a batch of experiences from replay memory.
  - Ensures there are enough experiences to sample a batch.
  - Samples a batch of experiences.
  - Converts experiences to PyTorch tensors and moves them to the appropriate device.

```python
        q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
```
- **q_values**: Q-values predicted by the primary model for the selected actions.
- **max_next_q_values**: Maximum Q-values predicted by the target model for the next states.
- **target_q_values**: Target Q-values computed using the Bellman equation.

```python
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```
- **loss**: Mean Squared Error (MSE) loss between predicted and target Q-values.
  - Computes the loss.
  - Clears previous gradients.
  - Backpropagates the loss.
  - Updates the model weights using the optimizer.

```python
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f'Episode: {self.step_counter // batch_size}, Epsilon: {self.epsilon}')
```
- **epsilon decay**: Reduces the exploration rate after each training step, but ensures it doesn't go below the minimum value.
- **logging**: Logs the current episode and epsilon value for tracking progress.

# Primary Q-Network

- **Purpose**: The primary Q-network, often just called the Q-network, is used to predict Q-values for each action given a state. It is the network that is being trained and directly used for making decisions during the learning process.
- **Training**: This network is updated frequently through backpropagation using the experiences sampled from the replay memory.
- **Usage**: During the agent's interaction with the environment, the primary Q-network is used to select actions, especially during exploitation (when not exploring randomly).

# Target Network

- **Purpose**: The target network is used to provide stable target Q-values for the updates to the primary Q-network. Its weights are copied from the primary Q-network at regular intervals, which helps in stabilizing the training process.
- **Stability**: By keeping the target network fixed for a number of steps, it reduces the risk of oscillations and divergence during training. This is because the target values (provided by the target network) change less frequently, providing a stable reference for the primary Q-network to learn from.
- **Usage**: When computing the target Q-values in the Bellman equation during training, the target network is used to estimate the future rewards.

### Detailed Example

To understand the interaction between these two networks, consider the following detailed steps in the DQN algorithm:

1. **Action Selection**:
   - The primary Q-network is used to select the best action based on the current state (if not exploring randomly).

2. **Storing Experience**:
   - The agent interacts with the environment, takes actions, and stores the resulting experiences (state, action, reward, next state, done) in the replay memory.

3. **Batch Sampling**:
   - A batch of experiences is randomly sampled from the replay memory for training.

4. **Target Q-Value Calculation**:
   - For each experience in the batch, the target Q-value is calculated using the target network. The target Q-value for a given experience is computed as:
     \[
     \text{target\_q\_value} = \text{reward} + \gamma \times \max(\text{target\_network}(next\_state)) \times (1 - \text{done})
     \]
   - Here, \(\gamma\) is the discount factor, and \(\text{done}\) indicates whether the episode has ended.

5. **Training the Primary Q-Network**:
   - The primary Q-network is trained to minimize the difference (loss) between its predicted Q-values and the target Q-values. The loss is typically computed using Mean Squared Error (MSE):
     \[
     \text{loss} = \text{MSE}(\text{primary\_network}(state, action), \text{target\_q\_value})
     \]
   - Backpropagation is used to update the weights of the primary Q-network.

6. **Updating the Target Network**:
   - Periodically (every \(N\) steps, as defined by `update_target_frequency`), the weights of the target network are updated to match those of the primary Q-network. This step is performed to ensure the stability of the target values used for training.

### Code Implementation

Here’s how these concepts are implemented in the provided `DQNAgent` class:

```python
class DQNAgent:
    def __init__(self, state_size, action_size, update_target_frequency=10):
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
        self.target_model.load_state_dict(self.model.state_dict())
```

- **Primary Q-Network**: `self.model` is the primary Q-network that is used for action selection and is trained using backpropagation.
- **Target Network**: `self.target_model` is the target network that provides stable Q-value targets for training the primary network.
- **update_target_model**: This method copies the weights from the primary Q-network to the target network, ensuring the target network is periodically updated.
