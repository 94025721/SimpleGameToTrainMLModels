import math

from entities.movement_strategies.movement_strategie import MovementStrategy


class CircularMovement(MovementStrategy):
    def __init__(self):
        self.angle = 0

    def move(self, enemy):
        self.angle += enemy.speed
        enemy.x = enemy.x_min + (enemy.x_max - enemy.x_min) / 2 + (enemy.x_max - enemy.x_min) / 2 * math.cos(self.angle)
        enemy.y = enemy.y_min + (enemy.y_max - enemy.y_min) / 2 + (enemy.y_max - enemy.y_min) / 2 * math.sin(self.angle)
        if self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi
