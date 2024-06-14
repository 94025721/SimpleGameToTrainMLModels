import math
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN


class GameEnvironment:
    """
    Wrapper for the game environment to interface with the DQN agent.
    """

    def __init__(self, game):
        """
        Initialize the game environment with a game instance.

        Args:
            game (Game): The game instance.
        """
        self.game = game
        self.previous_distance = None
        self.previous_player_deaths = 0

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            list: The initial state of the environment.
        """
        self.game.reset()
        self.previous_distance = self.calculate_distance_to_target(self.game.player)
        return self.get_state()

    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            list: The current state including player and enemies' positions and velocities.
        """
        player = self.game.player
        state = [player.x, player.y, player.dx, player.dy]
        for enemy in self.game.levels[self.game.current_level_index].enemies:
            state.extend([enemy.x, enemy.y, enemy.dx, enemy.dy])
        return state

    def step(self, action):
        """
        Perform an action in the environment and return the next state, reward, and done flag.

        Args:
            action (int): The action to perform.

        Returns:
            tuple: Next state, reward, and done flag.
        """
        keys = {K_LEFT: False, K_RIGHT: False, K_UP: False, K_DOWN: False}
        if action == 0:
            keys[K_LEFT] = True
        elif action == 1:
            keys[K_RIGHT] = True
        elif action == 2:
            keys[K_UP] = True
        elif action == 3:
            keys[K_DOWN] = True

        self.game.player.dx, self.game.player.dy = 0, 0
        if keys[K_LEFT]:
            self.game.player.dx = -2
        if keys[K_RIGHT]:
            self.game.player.dx = 2
        if keys[K_UP]:
            self.game.player.dy = -2
        if keys[K_DOWN]:
            self.game.player.dy = 2

        self.game.update()
        next_state = self.get_state()

        reward, done = self.calculate_reward()
        if done or self.previous_player_deaths - self.game.player.player_deaths > 0:
            self.reset()  # Reset the game on death

        return next_state, reward, done

    def calculate_distance_to_target(self, player):
        """
        Calculate the Euclidean distance from the player to the target zone.

        Args:
            player (Player): The player instance.

        Returns:
            float: The distance to the target zone.
        """
        target = self.game.levels[self.game.current_level_index].target_zone
        return math.sqrt((player.x - target.x) ** 2 + (player.y - target.y) ** 2)

    def calculate_reward(self):
        """
        Calculate the reward for the current state.

        Returns:
            tuple: Reward and done flag.
        """
        player = self.game.player
        current_distance = self.calculate_distance_to_target(player)

        reward = -0.1  # Default small penalty to encourage movement
        done = False

        if player.isFinished:
            reward += 200
            done = True
        elif player.player_deaths - self.previous_player_deaths > 0:
            reward -= 100
            self.previous_player_deaths = player.player_deaths
        if self.previous_distance is not None:
            distance_reward = self.previous_distance - current_distance
            reward += distance_reward
        self.previous_distance = current_distance

        return reward, done

    def render(self):
        """
        Render the current state of the game.
        """
        self.game.game_observer.update()
