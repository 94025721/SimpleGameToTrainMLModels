import logging

import pygame
import sys

from managers.training_log_manager import TrainingLogManager
from settings import Settings
from ui.game_panel import GamePanel
from managers.game_manager import GameManager
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, QUIT, KEYDOWN, KEYUP
from dqn.dqn_agent import DQNAgent
from dqn.game_environment import GameEnvironment


def main(mode):
    """
    Main function to run the game in different modes.

    Args:
        mode (str): The mode to run the game in. Choices are 'play', 'train', and 'plot'.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    initialize_pygame()
    screen = pygame.display.set_mode((Settings.GAME_WIDTH, Settings.GAME_HEIGHT))
    pygame.display.set_caption("Simple 2D Game")

    game_panel, game = initialize_game(screen)

    if mode == 'play':
        manual_play(game, game_panel)
    elif mode == 'train':
        TrainingLogManager.initialize_new_log_file()
        train_dqn(game, episodes=1000, batch_size=16)
    elif mode == 'plot':
        TrainingLogManager.plot_training_progress()
    else:
        raise ValueError(f"Invalid mode: {mode}")


def initialize_pygame():
    """
    Initialize the pygame library and set up logging.
    """
    pygame.init()
    logging.basicConfig(level=logging.INFO)


def initialize_game(screen):
    """
    Initialize the game panel and game manager.

    Args:
        screen (pygame.Surface): The screen surface for rendering the game.

    Returns:
        tuple: A tuple containing the initialized game panel and game manager.
    """
    game_panel = GamePanel(screen)
    game = GameManager(game_panel)
    game_panel.set_game(game)
    return game_panel, game


def manual_play(game, game_panel):
    """
    Run the game in manual play mode, allowing the player to control the character with the keyboard.

    Args:
        game (GameManager): The game manager instance.
        game_panel (GamePanel): The game panel instance for rendering the game.
    """
    keys = {K_LEFT: False, K_RIGHT: False, K_UP: False, K_DOWN: False}
    running = True

    while running:
        handle_events(keys)
        update_player_movement(game, keys)
        game.update()
        game_panel.update()
        game.clock.tick(60)


def handle_events(keys):
    """
    Handle pygame events, updating the keys dictionary and processing quit events.

    Args:
        keys (dict): Dictionary mapping key constants to their pressed state.
    """
    for event in pygame.event.get():
        if event.type == QUIT:
            process_quit_event()
        elif event.type in (KEYDOWN, KEYUP):
            handle_key_event(keys, event)


def handle_key_event(keys, event):
    """
    Handle key down and key up events, updating the keys dictionary.

    Args:
        keys (dict): Dictionary mapping key constants to their pressed state.
        event (pygame.event.Event): The key event to handle.
    """
    if event.key in keys:
        keys[event.key] = (event.type == KEYDOWN)


def update_player_movement(game, keys):
    """
    Update the player's movement based on the current state of the keys dictionary.

    Args:
        game (GameManager): The game manager instance.
        keys (dict): Dictionary mapping key constants to their pressed state.
    """
    player = game.player
    player.dx, player.dy = 0, 0

    if keys[K_LEFT]:
        player.dx = -2
    if keys[K_RIGHT]:
        player.dx = 2
    if keys[K_UP]:
        player.dy = -2
    if keys[K_DOWN]:
        player.dy = 2


def train_dqn(game, episodes, batch_size, max_steps_per_episode=500):
    """
    Train a DQN agent on the game environment.

    Args:
        game (GameManager): The game manager instance.
        episodes (int): Number of episodes to train the agent.
        batch_size (int): Batch size for training the agent.
        max_steps_per_episode (int): Maximum number of steps per episode. Default is 500.
    """
    state_size = 4 + 4 * len(game.levels[game.current_level_index].enemies)
    action_size = 4  # [left, right, up, down]
    agent = DQNAgent(state_size, action_size)
    env = GameEnvironment(game)
    model_path = "dqn_model.pth"
    memory_path = "replay_memory.pkl"
    # agent.load(model_path, memory_path)  # Uncomment this line to load the previous model and memory

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for time_step in range(max_steps_per_episode):
            if process_quit_event():
                return

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            env.render()

            if done:
                break

        TrainingLogManager.record_episode_metrics(e, total_reward, agent.epsilon, game.finish_count)
        logging.info(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        agent.replay(batch_size)

    agent.save(model_path, memory_path)


def process_quit_event():
    """
    Process quit events, exiting the game if a quit event is detected.

    Returns:
        bool: True if a quit event was processed, False otherwise.
    """
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    return False
