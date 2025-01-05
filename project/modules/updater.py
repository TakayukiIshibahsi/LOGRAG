from vectordb import Memory

class Updater:
    def __init__(self, memory: Memory):
        self.memory = memory

    def update(self, key, value):
        self.memory.set(key, value)
        