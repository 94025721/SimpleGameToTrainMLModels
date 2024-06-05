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
        self.initial_state = None

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

    def set_target_zone(self, target_zone):
        self.target_zone = target_zone

    def save_initial_state(self):
        self.initial_state = {
            "coins": [coin.clone() for coin in self.coins],
            "enemies": [enemy.clone() for enemy in self.enemies],
            "walls": [wall.clone() for wall in self.walls],
            "target_zone": self.target_zone.clone() if self.target_zone else None,
            "spawn_x": self.spawn_x,
            "spawn_y": self.spawn_y
        }

    def reset(self):
        self.coins = [coin.clone() for coin in self.initial_state["coins"]]
        self.enemies = [enemy.clone() for enemy in self.initial_state["enemies"]]
        self.walls = [wall.clone() for wall in self.initial_state["walls"]]
        self.target_zone = self.initial_state["target_zone"].clone() if self.initial_state["target_zone"] else None
        self.spawn_x = self.initial_state["spawn_x"]
        self.spawn_y = self.initial_state["spawn_y"]
