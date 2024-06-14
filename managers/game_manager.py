import pygame
from levels.level_loader import LevelLoader
from managers.collision_manager import CollisionManager
from entities.player import Player


class Game:
    RUNNING, PAUSED, GAME_OVER = range(3)

    def __init__(self, game_observer):
        self.game_observer = game_observer
        self.player = Player()
        self.finish_count = 0
        self.current_level_index = 0
        self.game_state = self.RUNNING
        self.collision_manager = CollisionManager(self)
        self.levels = []
        self.load_levels()

        self.clock = pygame.time.Clock()
        self.timer_event = pygame.USEREVENT + 1
        pygame.time.set_timer(self.timer_event, 16)

    def load_levels(self):
        try:
            self.levels = LevelLoader.load_all_levels()
            self.respawn_player()
        except Exception as e:
            print(e)

    def respawn_player(self):
        if self.levels:
            current_level = self.levels[self.current_level_index]
            self.player.isFinished = False
            self.player.respawn(current_level.spawn_x, current_level.spawn_y)

    def update(self):
        if self.game_state == self.RUNNING:
            current_level = self.levels[self.current_level_index]
            self.collision_manager.handle_player_movement(self.player, current_level)
            self.collision_manager.handle_enemy_movement(self.player, current_level)
            self.game_observer.update()

    def add_level(self, level):
        self.levels.append(level)

    def next_level(self):
        if self.current_level_index < len(self.levels) - 1:
            self.current_level_index += 1
            self.respawn_player()

    def reset(self):
        self.levels[self.current_level_index].reset()
        self.respawn_player()
        self.player.times_finished = False

    def increment_finish_count(self):
        self.finish_count += 1
