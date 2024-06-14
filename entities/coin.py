from pygame import Rect
from .entity import Entity


class Coin(Entity):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def get_bounds(self):
        return Rect(self.x, self.y, 10, 10)
