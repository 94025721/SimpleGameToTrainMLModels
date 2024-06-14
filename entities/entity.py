import copy

class Entity:
    """Base class for all game entities."""

    def get_bounds(self):
        """Returns the bounding box of the entity.

        Returns:
            Rect: The rectangular bounds of the entity.
        """
        pass

    def clone(self):
        """Creates a deep copy of the entity.

        Returns:
            Entity: A deep copy of the entity.
        """
        return copy.deepcopy(self)