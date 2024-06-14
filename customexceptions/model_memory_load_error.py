class ModelMemoryLoadError(Exception):
    def __init__(self, message="No saved model/memory found, starting from scratch."):
        self.message = message
        super().__init__(self.message)