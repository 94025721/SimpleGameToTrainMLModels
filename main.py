import pygame
import sys

from settings import Settings
from ui.game_panel import GamePanel
from managers.game_manager import Game
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, QUIT, KEYDOWN, KEYUP
from dqn.dqn_agent import DQNAgent
from dqn.game_environment import GameEnvironment
import argparse


def main(mode):
    pygame.init()
    screen = pygame.display.set_mode((Settings.GAME_WIDTH, Settings.GAME_HEIGHT))
    pygame.display.set_caption("Simple 2D Game")
    game_panel = GamePanel(screen)
    game = Game(game_panel)
    game_panel.set_game(game)

    if mode == 'play':
        manual_play(game, game_panel)
    elif mode == 'train':
        train_dqn(game, episodes=1000, batch_size=16)

    pygame.quit()
    sys.exit()


def manual_play(game, game_panel):
    keys = {
        K_LEFT: False,
        K_RIGHT: False,
        K_UP: False,
        K_DOWN: False
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN or event.type == KEYUP:
                handle_key_event(keys, event)

        update_player_movement(game, keys)
        game.update()
        game_panel.update()
        game.clock.tick(60)


def handle_key_event(keys, event):
    if event.type == KEYDOWN and event.key in keys:
        keys[event.key] = True
    elif event.type == KEYUP and event.key in keys:
        keys[event.key] = False


def update_player_movement(game, keys):
    player = game.player
    player.dx = 0
    player.dy = 0
    if keys[K_LEFT]:
        player.dx = -2
    if keys[K_RIGHT]:
        player.dx = 2
    if keys[K_UP]:
        player.dy = -2
    if keys[K_DOWN]:
        player.dy = 2


def train_dqn(game, episodes, batch_size):
    state_size = 4  # [player.x, player.y, player.dx, player.dy]
    action_size = 4  # [left, right, up, down]
    agent = DQNAgent(state_size, action_size)
    env = GameEnvironment(game)
    model_path = "dqn_model.pth"
    memory_path = "replay_memory.pkl"
    # agent.load(model_path, memory_path)

    for e in range(episodes):
        state = env.reset()
        for time_step in range(500):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, Score: {time_step}, Epsilon: {agent.epsilon:.2}")
                break
            agent.replay(batch_size)
            env.render()

        agent.save(model_path, memory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple 2D Game")
    parser.add_argument('--mode', choices=['play', 'train'], required=True, help="Mode to run the game in")
    args = parser.parse_args()

    try:
        main(args.mode)
    except KeyboardInterrupt:
        print("Game terminated by user")
