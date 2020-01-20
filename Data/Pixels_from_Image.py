import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class PixelsFromImage:
    def __init__(self, image, nb_elements=2000):
        self.data = None
        self.image = image
        self.nb_elements = nb_elements
        self.create()

    def create(self):
        size = np.flip(self.image.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        # print(size)
        pixels = np.array(self.image.getdata(), 'uint8')
        # print(pixels.shape)
        self.data = np.array(pixels, 'double') / 255
        np.random.shuffle(self.data)
        self.data.resize((self.nb_elements, 3))
        # print(self.data.shape)
        # print(self.data)

    def get_data(self):
        return self.data

    def display(self):
        return plt.imshow(self.image)


if __name__ == '__main__':
    PixelsFromImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"))
    print("Yay")
