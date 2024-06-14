from pygame import Rect
from .entity import Entity


class Player(Entity):
    """
    Represents the player character in the game.
    """

    def __init__(self, x=0, y=0):
        """
        Initialize the Player with default or specified coordinates.

        Args:
            x (int): Initial x-coordinate of the player. Default is 0.
            y (int): Initial y-coordinate of the player. Default is 0.
        """
        self.x = x
        self.y = y
        self.isFinished = False
        self.dx = 0
        self.dy = 0
        self.prev_x = 0
        self.prev_y = 0
        self.coins_collected = 0
        self.player_deaths = 0
        self.width = 20
        self.height = 20

    def move(self):
        """
        Move the player based on its current velocity.
        """
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.dx
        self.y += self.dy

    def undo_move(self):
        """
        Undo the player's last move, reverting to the previous position.
        Used in the collision manager {@link /managers/collision_manager.py}.
        """
        self.x = self.prev_x
        self.y = self.prev_y

    def get_bounds(self):
        """
        Get the rectangular bounds of the player.

        Returns:
            pygame.Rect: The rectangular bounds of the player.
        """
        return Rect(self.x, self.y, self.width, self.height)

    def respawn(self, x, y):
        """
        Respawn the player at a new position.

        Args:
            x (int): New x-coordinate of the player.
            y (int): New y-coordinate of the player.
        """
        self.x = x
        self.y = y

    def increment_player_deaths(self):
        """
        Increment the player's death count by one.
        """
        self.player_deaths += 1

    def increment_coins_collected(self):
        """
        Increment the player's coin collection count by one.
        """
        self.coins_collected += 1

    def get_state(self):
        """
        Get the current state of the player.

        Returns:
            list: A list containing the player's x, y, dx, and dy.
        """
        return [self.x, self.y, self.dx, self.dy]
