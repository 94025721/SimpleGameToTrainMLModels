from entities.movement_strategies.movement_strategie import MovementStrategy


class DiagonalMovement(MovementStrategy):
    def move(self, enemy):
        enemy.x += enemy.speed
        enemy.y += enemy.speed
        if enemy.x > enemy.x_max or enemy.x < enemy.x_min or enemy.y > enemy.y_max or enemy.y < enemy.y_min:
            enemy.speed = -enemy.speed
