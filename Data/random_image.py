import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Code.Parameters import Parameters


class RandomImage:
    def __init__(self, image, parameters):
        self.data = None
        self.image = image
        self.size = None
        self.pictures_dim = parameters["pictures_dim"]
        self.color = False
        self.nb_pictures = None
        self.sliding_window()

    def sliding_window(self):
        self.data = []
        self.size = np.flip(self.image.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        # print(self.size)
        pixels = np.array(self.image.getdata(), 'uint8')
        # print(pixels.shape)
        if len(pixels.shape) == 2:  # File has RGB colours
            self.size[1] *= 3
            self.pictures_dim[1] *= 3
            self.color = True
        pixels = pixels.reshape(self.size)
        for i in range(0, self.size[0] - self.pictures_dim[0], 1):
            for j in range(0, self.size[1] - self.pictures_dim[1], 3):
                self.data.append(np.array(pixels[i:i+self.pictures_dim[0], j:j+self.pictures_dim[1]]).flatten())
        self.data = np.array(self.data) / 255

    def reconstruct(self, data, size=None):
        size = self.size
        dim = self.pictures_dim
        if self.color:
            size = [self.size[0], self.size[1] // 3, 3]
            dim = [self.pictures_dim[0], self.pictures_dim[1] // 3, 3]
        pixels = np.zeros(size)

        count = 0
        for i in range(0, size[0] - dim[0], 5):
            for j in range(0, size[1] - dim[1], 5):
                # print(i,j)
                current = data[count]
                count += 1
                # print(current)
                current = current.reshape(dim)
                # current = np.rot90(current, 2, (0, 1))
                # current = np.flip(current, (0,1))
                # print(pixels.shape)
                # print(current.shape)
                # print(self.pictures_dim)
                # print(self.size)
                pixels[i:i+dim[1], j:j+dim[1], :] += current
        pixels *= (255/4)
        pixels = np.array(pixels, 'uint8')
        return pixels

    def get_data(self, size):
        np.random.shuffle(self.data)
        return self.data[0:size]

    def display(self):
        return plt.imshow(self.image)


if __name__ == '__main__':
    bkg = Image.open("Data/color_test.png")
    pictures_dim = [10, 10]
    parameters = Parameters({"pictures_dim": pictures_dim})
    data = RandomImage(bkg, parameters)