import math
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN


class GameEnvironment:
    def __init__(self, game):
        self.game = game
        self.previous_distance = None
        self.previous_player_deaths = 0

    def reset(self):
        self.game.reset()
        self.previous_distance = self.calculate_distance_to_target(self.game.player)
        return self.get_state()

    def get_state(self):
        player = self.game.player
        state = [player.x, player.y, player.dx, player.dy]
        for enemy in self.game.levels[self.game.current_level_index].enemies:
            state.extend([enemy.x, enemy.y, enemy.dx, enemy.dy])
        return state

    def step(self, action):
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
        target = self.game.levels[self.game.current_level_index].target_zone
        return math.sqrt((player.x - target.x) ** 2 + (player.y - target.y) ** 2)

    def calculate_reward(self):
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
        self.game.game_observer.update()
