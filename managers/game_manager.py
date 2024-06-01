import pygame
from levels.level_loader import LevelLoader
from managers.collision_manager import CollisionManager
from entities.player import Player


class Game:
    RUNNING, PAUSED, GAME_OVER = range(3)

    def __init__(self, game_observer):
        self.game_observer = game_observer
        self.player = Player()
        self.current_level_index = 0
        self.game_state = self.RUNNING
        self.collision_manager = CollisionManager()
        self.levels = []
        self.load_levels()

        self.clock = pygame.time.Clock()
        self.timer_event = pygame.USEREVENT + 1
        pygame.time.set_timer(self.timer_event, 16)

    def load_levels(self):
        try:
            levels = LevelLoader.load_all_levels('levels.json')
            self.levels.extend(levels)
            if self.levels:
                self.player.respawn(self.levels[self.current_level_index].spawn_x,
                                    self.levels[self.current_level_index].spawn_y)
        except Exception as e:
            print(e)

    def update(self):
        if self.game_state == self.RUNNING:
            current_level = self.levels[self.current_level_index]
            self.collision_manager.handle_player_movement(self.player, current_level)
            self.collision_manager.handle_enemy_movement(self.player, current_level)
            self.collision_manager.handle_coin_collection(self.player, current_level)
            if self.player.finished:
                self.game_state = self.GAME_OVER
            self.game_observer.update()

    def add_level(self, level):
        self.levels.append(level)

    def next_level(self):
        if self.current_level_index < len(self.levels) - 1:
            self.current_level_index += 1

    def reset(self):
        self.current_level_index = 0
        self.player.respawn(self.levels[self.current_level_index].spawn_x,
                            self.levels[self.current_level_index].spawn_y)
        self.player.finished = False

    def is_done(self):
        return self.player.finished
