class CollisionManager:
    """
    Manages collision detection and handling for the game.
    """

    def __init__(self, game):
        """
        Initialize the collision manager with the game instance.

        Args:
            game (Game): The game instance.
        """
        self.game = game

    def handle_player_movement(self, player, current_level):
        """
        Handle the player's movement and collisions with walls and the target zone.

        Args:
            player (Player): The player instance.
            current_level (Level): The current level instance.
        """
        player.move()
        for wall in current_level.walls:
            if player.get_bounds().colliderect(wall.get_bounds()):
                player.undo_move()

        if player.get_bounds().colliderect(current_level.target_zone.get_bounds()):
            player.isFinished = True
            self.game.increment_finish_count()
            self.game.reset()

    def handle_enemy_movement(self, player, current_level):
        """
        Handle the enemies' movement and collisions with the player.

        Args:
            player (Player): The player instance.
            current_level (Level): The current level instance.
        """
        for enemy in current_level.enemies:
            enemy.move()
            if enemy.get_bounds().colliderect(player.get_bounds()):
                self.handle_player_death(player)

    def handle_coin_collection(self, player, current_level):
        """
        Handle the collection of coins by the player.

        Args:
            player (Player): The player instance.
            current_level (Level): The current level instance.
        """
        coins_to_remove = []
        for coin in current_level.coins:
            if player.get_bounds().colliderect(coin.get_bounds()):
                player.increment_coins_collected()
                coins_to_remove.append(coin)
        for coin in coins_to_remove:
            current_level.coins.remove(coin)

    def handle_player_death(self, player):
        """
        Handle the player's death and reset the game.

        Args:
            player (Player): The player instance.
        """
        player.increment_player_deaths()
        self.game.reset()

