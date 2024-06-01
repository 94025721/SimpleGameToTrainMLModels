from .entity import Entity
from pygame import Rect

class Wall(Entity):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_bounds(self):
        return Rect(self.x, self.y, self.width, self.height)
