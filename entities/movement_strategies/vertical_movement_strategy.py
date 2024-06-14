from entities.movement_strategies.movement_strategie import MovementStrategy


class VerticalMovement(MovementStrategy):
    def move(self, enemy):
        enemy.y += enemy.speed
        if enemy.y > enemy.y_max or enemy.y < enemy.y_min:
            enemy.speed = -enemy.speed
