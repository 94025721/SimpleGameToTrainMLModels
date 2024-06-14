# Simple 2D Game with DQN Agent

## Overview

This project is a simple 2D game implemented using Pygame, with a Deep Q-Network (DQN) agent for training and playing the game autonomously. The game involves navigating a player through levels, collecting coins, avoiding enemies, and reaching a target zone.

## Features

- **Player Movement**: Control the player using keyboard inputs or an AI agent.
- **Enemy Movement**: Enemies move in predefined patterns.
- **Coin Collection**: Collect coins scattered throughout the level.
- **Target Zone**: Reach the target zone to finish the level.
- **Level Loading**: Levels are loaded from JSON files.
- **DQN Agent**: Train a DQN agent to play the game using reinforcement learning.
- **Replay Memory**: Store and sample experiences for training the DQN agent.
- **Game Modes**: Play manually, train the DQN agent, or plot training results.

## Usage

### Manual Play

To play the game manually, run:

```sh
python main.py --mode play
```

### Train DQN Agent

To train the DQN agent, run:

```sh
python main.py --mode train
```

### Plot Training Results

To plot the scores and epsilon values after training, run:

```sh
python main.py --mode plot
```

## Game Controls

- **Left Arrow**: Move left
- **Right Arrow**: Move right
- **Up Arrow**: Move up
- **Down Arrow**: Move down

## Project Structure

- `entities/`: Contains entity classes such as `Player`, `Enemy`, `Coin`, `Wall`, and movement strategies.
- `levels/`: Contains `Level` and `LevelLoader` classes for managing game levels.
- `managers/`: Contains manager classes such as `CollisionManager` and `GameManager`.
- `dqn/`: Contains DQN-related classes such as `DQNAgent`, `DQN`, and `ReplayMemory`.
- `settings.py`: Contains game settings and configurations.
- `main.py`: Main script to run the game in different modes.
- `resources/levels/`: Directory containing level JSON files.

## Level JSON Format

Each level is defined in a JSON file with the following structure:

```json
{
  "spawnX": 50,
  "spawnY": 50,
  "targetZone": {
    "x": 200,
    "y": 200,
    "width": 40,
    "height": 40
  },
  "enemies": [
    {
      "x": 100,
      "y": 100,
      "x_max": 300,
      "y_max": 300,
      "x_min": 50,
      "y_min": 50,
      "speed": 2,
      "movementType": "horizontal"
    }
  ],
  "walls": [
    {
      "x": 0,
      "y": 0,
      "width": 400,
      "height": 20
    }
  ],
  "coins": [
    {
      "x": 150,
      "y": 150,
      "width": 10,
      "height": 10
    }
  ]
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Pygame](https://www.pygame.org/)
- [PyTorch](https://pytorch.org/)

## Contact

For any questions or inquiries, please contact [nl.degraaff.n@proton.me].
