from entities.movement_strategies.movement_strategie import MovementStrategy


class HorizontalMovement(MovementStrategy):
    def move(self, enemy):
        enemy.x += enemy.speed
        if enemy.x > enemy.x_max or enemy.x < enemy.x_min:
            enemy.speed = -enemy.speed
