# 2D Game with DQN and Q-learning

This project is a simple 2D game developed to learn and experiment with Deep Q-Network (DQN) and Q-learning algorithms. The player must dodge enemies, collect coins, and reach the target zone to complete levels. The game can be run in different modes to either train the DQN agent, play the game manually, or plot the training data.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Pygame

Install the required Python packages using pip:

```bash
pip install torch numpy matplotlib pygame
```

### Running the Game

The game can be run with three different modes: `train`, `play`, or `plot`.

- **Train**: Trains the DQN agent.
- **Play**: Play the game manually.
- **Plot**: Plot the training data saved during the training sessions.

#### Train the Agent

To train the DQN agent, run the following command:

```bash
python3 main.py train
```

This will train the agent and save the training data in a JSON file for later analysis. Each training session generates a unique log file to help study the performance of different agent training configurations.

#### Play the Game

To play the game manually, run:

```bash
python3 main.py play
```

#### Plot Training Data

To plot the training data and visualize the agent's performance over time, run:

```bash
python3 main.py plot
```

## Creating Levels

Levels are defined in JSON files and can be loaded into the game using the level loader. Here is an example of how to create a level JSON file:

```json
{
    "spawnX": 50,
    "spawnY": 50,
    "targetZone": {
        "x": 300,
        "y": 300,
        "width": 20,
        "height": 20
    },
    "enemies": [
        {
            "x": 100,
            "y": 100,
            "x_min": 50,
            "x_max": 150,
            "y_min": 50,
            "y_max": 150,
            "speed": 2,
            "movementType": "horizontal"
        }
    ],
    "walls": [
        {"x": 0, "y": 0, "width": 20, "height": 400},
        {"x": 0, "y": 0, "width": 400, "height": 20}
    ],
    "coins": [
        {"x": 200, "y": 200}
    ]
}
```

### Movement Strategies for Enemies

Enemies can have different movement strategies:

- **Vertical**: Moves up and down.
- **Horizontal**: Moves left and right.
- **Diagonal**: Moves diagonally.
- **Circular**: Moves in a circular path.

Specify the movement type in the `movementType` field for each enemy in the level JSON file.

## Contributing

Feel free to fork this repository, create merge requests, and contribute. Improvements and suggestions are highly appreciated.

## Acknowledgements

This game is inspired by the "World's Hardest Game".

---

Happy gaming and learning!
