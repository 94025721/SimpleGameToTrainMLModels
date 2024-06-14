import pygame
from .entity import Entity


class TargetZone(Entity):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_bounds(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


