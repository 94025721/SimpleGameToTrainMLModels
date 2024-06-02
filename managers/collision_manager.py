class CollisionManager:
    def handle_player_movement(self, player, current_level):
        player.move()
        for wall in current_level.walls:
            if player.get_bounds().colliderect(wall.get_bounds()):
                player.undo_move()

        if player.get_bounds().colliderect(current_level.target_zone.get_bounds()):
            player.increment_times_finished()
            player.respawn(current_level.spawn_x, current_level.spawn_y)

    def handle_enemy_movement(self, player, current_level):
        for enemy in current_level.enemies:
            enemy.move()
            if enemy.get_bounds().colliderect(player.get_bounds()):
                self.handle_player_death(player, current_level)

    def handle_coin_collection(self, player, current_level):
        coins_to_remove = []
        for coin in current_level.coins:
            if player.get_bounds().colliderect(coin.get_bounds()):
                player.increment_coins_collected()
                coins_to_remove.append(coin)
        for coin in coins_to_remove:
            current_level.coins.remove(coin)

    def handle_player_death(self, player, current_level):
        player.respawn(current_level.spawn_x, current_level.spawn_y)
        player.increment_player_deaths()
