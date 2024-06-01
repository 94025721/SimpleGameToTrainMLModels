from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN


class GameEnvironment:
    def __init__(self, game):
        self.game = game

    def reset(self):
        self.game.reset()
        return self.get_state()

    def get_state(self):
        player = self.game.player
        return [player.x, player.y, player.dx, player.dy]

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
        reward = 0
        done = self.game.player.finished
        return next_state, reward, done

    def render(self):
        self.game.game_observer.update()
