from .entity import Entity
from pygame import Rect
from settings import Settings


class Enemy(Entity):
    def __init__(self, x, y, x_min, x_max, y_min, y_max, speed, movement_strategy):
        self.dy = 0
        self.dx = 0
        self.width = Settings.ENEMY_WIDTH
        self.height = Settings.ENEMY_HEIGHT
        self.x = x
        self.y = y
        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.speed = speed
        self.movement_strategy = movement_strategy

    def move(self):
        previous_x, previous_y = self.x, self.y
        self.movement_strategy.move(self)
        self.dx = self.x - previous_x
        self.dy = self.y - previous_y

    def undo_move(self):
        pass

    def get_bounds(self):
        return Rect(self.x, self.y, self.width, self.height)
