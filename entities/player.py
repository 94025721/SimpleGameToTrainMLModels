from pygame import Rect
from .entity import Entity


class Player(Entity):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.times_finished = 0
        self.dx = 0
        self.dy = 0
        self.prev_x = 0
        self.prev_y = 0
        self.coins_collected = 0
        self.player_deaths = 0
        self.width = 20
        self.height = 20

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.dx
        self.y += self.dy

    def undo_move(self):
        self.x = self.prev_x
        self.y = self.prev_y

    def get_bounds(self):
        return Rect(self.x, self.y, self.width, self.height)

    def respawn(self, x, y):
        self.x = x
        self.y = y

    def increment_player_deaths(self):
        self.player_deaths += 1
        print("Player deaths:", self.player_deaths)

    def increment_times_finished(self):
        self.times_finished += 1
        print("Times finished:", self.times_finished)

    def increment_coins_collected(self):
        self.coins_collected += 1
        print("Coins collected:", self.coins_collected)

    def get_state(self):
        return [self.x, self.y, self.dx, self.dy]
