import copy

class Entity:
    def get_bounds(self):
        pass

    def clone(self):
        return copy.deepcopy(self)