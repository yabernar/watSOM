import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class GenerateFromShape:
    def __init__(self, image, nb_elements):
        self.data = []
        self.image = image
        self.nb_elements = nb_elements
        self.create()

    def create(self):
        size = np.flip(self.image.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        # print(size)
        pixels = np.array(self.image.getdata(), 'uint8')
        pixels = pixels.reshape(size)
        print(pixels)

        for i in range(self.nb_elements):
            elem = np.random.random(2)
            while pixels[tuple(np.multiply(elem, size).astype(int))] <= 20:
                elem = np.random.random(2)
            self.data.append(elem)
        self.data = np.asarray(self.data)

    def get_data(self):
        return self.data

    def display(self):
        return plt.plot(self.data[:, 0], self.data[:, 1], 'r+')


if __name__ == '__main__':
    g = GenerateFromShape(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/cat-silhouette.png"), 2000)
    plot = [g.display()]
    plt.draw()
    plt.waitforbuttonpress()
