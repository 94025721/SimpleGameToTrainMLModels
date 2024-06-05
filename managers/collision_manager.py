class CollisionManager:
    def __init__(self, game):
        self.game = game

    def handle_player_movement(self, player, current_level):
        player.move()
        for wall in current_level.walls:
            if player.get_bounds().colliderect(wall.get_bounds()):
                player.undo_move()

        if player.get_bounds().colliderect(current_level.target_zone.get_bounds()):
            player.isFinished = True
            self.game.reset()

    def handle_enemy_movement(self, player, current_level):
        for enemy in current_level.enemies:
            enemy.move()
            if enemy.get_bounds().colliderect(player.get_bounds()):
                self.handle_player_death(player)

    def handle_coin_collection(self, player, current_level):
        coins_to_remove = []
        for coin in current_level.coins:
            if player.get_bounds().colliderect(coin.get_bounds()):
                player.increment_coins_collected()
                coins_to_remove.append(coin)
        for coin in coins_to_remove:
            current_level.coins.remove(coin)

    def handle_player_death(self, player):
        player.increment_player_deaths()
        self.game.reset()
