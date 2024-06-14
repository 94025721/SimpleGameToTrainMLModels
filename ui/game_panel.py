import pygame


class GamePanel:
    def __init__(self, screen):
        self.screen = screen
        self.game = None

    def set_game(self, game):
        self.game = game

    def draw_game(self):
        self.screen.fill((255, 255, 255))
        if not self.game:
            return

        player = self.game.player
        current_level = self.game.levels[self.game.current_level_index]

        pygame.draw.rect(self.screen, (0, 255, 0), current_level.target_zone.get_bounds())
        pygame.draw.rect(self.screen, (255, 0, 0), player.get_bounds())

        for enemy in current_level.enemies:
            pygame.draw.rect(self.screen, (0, 0, 255), enemy.get_bounds())

        for wall in current_level.walls:
            pygame.draw.rect(self.screen, (128, 128, 128), wall.get_bounds())

        for coin in current_level.coins:
            pygame.draw.ellipse(self.screen, (255, 255, 0), coin.get_bounds())

    def update(self):
        self.draw_game()
        pygame.display.flip()
