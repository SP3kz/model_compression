import numpy as np

class PhotonicMemory:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size)

    def store_data(self, index, value):
        self.data[index] = value

    def retrieve_data(self, index):
        return self.data[index]

# Example usage
storage = PhotonicMemory(100)
storage.store_data(10, 0.75)
print(storage.retrieve_data(10))
