from entities.wall import Wall
from settings import Settings


class Level:
    def __init__(self):
        self.coins = []
        self.enemies = []
        self.walls = []
        self.target_zone = None
        self.spawn_x = 50
        self.spawn_y = 50
        self.create_surrounding_walls()

    def create_surrounding_walls(self):
        game_width = Settings.GAME_WIDTH
        game_height = Settings.GAME_HEIGHT

        self.walls.append(Wall(0, 0, game_width, 20))
        self.walls.append(Wall(0, 0, 20, game_height))
        self.walls.append(Wall(0, game_height - 20, game_width, 20))
        self.walls.append(Wall(game_width - 20, 0, 20, game_height))

    def add_coin(self, coin):
        self.coins.append(coin)

    def add_enemy(self, enemy):
        self.enemies.append(enemy)

    def add_wall(self, wall):
        self.walls.append(wall)
