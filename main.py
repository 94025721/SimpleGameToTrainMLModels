import pygame
import sys
from settings import Settings
from ui.game_panel import GamePanel
from managers.game_manager import Game
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, QUIT, KEYDOWN, KEYUP


def main():
    pygame.init()
    screen = pygame.display.set_mode((Settings.GAME_WIDTH, Settings.GAME_HEIGHT))
    pygame.display.set_caption("Simple 2D Game")
    game_panel = GamePanel(screen)
    game = Game(game_panel)
    game_panel.set_game(game)

    # Initialize key state dictionary
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

    pygame.quit()
    sys.exit()


def handle_key_event(keys, event):
    if event.type == KEYDOWN:
        if event.key in keys:
            keys[event.key] = True
    elif event.type == KEYUP:
        if event.key in keys:
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Game terminated by user")
