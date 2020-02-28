import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Code.Parameters import Parameters


class SlidingWindow:
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
        for i in range(0, self.size[0] - self.pictures_dim[0], 5):
            for j in range(0, self.size[1] - self.pictures_dim[1], 15):
                self.data.append(np.array(pixels[i:i+self.pictures_dim[0], j:j+self.pictures_dim[1]]))
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
                current = data[count]
                count += 1
                # print(current)
                current = current.reshape(dim)
                # current = np.flip(current, (0,1))
                # print(pixels.shape)
                # print(current.shape)
                # print(self.pictures_dim)
                # print(self.size)
                pixels[i:i+dim[1], j:j+dim[1], :] += current
        pixels *= (255/2)
        pixels = np.array(pixels, 'uint8')
        return pixels

    def get_data(self):
        return self.data

    def display(self):
        return plt.imshow(self.image)


if __name__ == '__main__':
    bkg = Image.open("Data/color_test.png")
    pictures_dim = [10, 10]
    parameters = Parameters({"pictures_dim": pictures_dim})
    data = SlidingWindow(bkg, parameters)


# def compress(self, som):
#     som_map = som.get_som_as_map()
#     pixels = []
#     dim = [self.real_picture_dim[0], self.real_picture_dim[1]//3, 3]
#     for i in range(len(self.data)):
#         w = som.winner(self.data[i])
#         pixels.append(som_map[w])
#     px2 = []
#     lst2 = ()
#     for i in range(self.nb_pictures[0]):
#         lst = ()
#         for j in range(self.nb_pictures[1]):
#             pixels[i*self.nb_pictures[1]+j] = pixels[i*self.nb_pictures[1]+j].reshape(dim)
#             lst = lst + (pixels[i*self.nb_pictures[1]+j],)
#         px2.append(np.concatenate(lst, axis=1))
#         lst2 += (px2[i],)
#     px = np.concatenate(lst2, axis=0)
#     px *= 255
#     px = np.array(px, 'uint8')
#     file = Image.fromarray(px)
#     return file
#
# def compress_channel(self, som, channel):
#     som_map = som.get_som_as_map()
#     pixels = []
#     for i in range(len(self.data)):
#         w = som.winner(self.data[i])
#         pixels.append(som_map[w])
#     px2 = []
#     lst2 = ()
#     for i in range(self.nb_pictures[0]):
#         lst = ()
#         for j in range(self.nb_pictures[1]):
#             pixels[i*self.nb_pictures[1]+j] = pixels[i*self.nb_pictures[1]+j].reshape(pictures_dim)
#             lst = lst + (pixels[i*self.nb_pictures[1]+j],)
#         px2.append(np.concatenate(lst, axis=1))
#         lst2 += (px2[i],)
#     px = np.concatenate(lst2, axis=0)
#     px *= 255
#     px = np.array(px, 'uint8')
#     file = Image.fromarray(px)
#     return file
#
# def display_som(self, som_list):
#     som_list = som_list * 255
#     px2 = []
#     lst2 = ()
#     dim = [self.real_picture_dim[0], self.real_picture_dim[1]//3, 3]
#     for y in range(neuron_nbr):
#         lst = ()
#         for x in range(neuron_nbr):
#             som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(dim)
#             lst = lst + (som_list[y * neuron_nbr + x],)
#         px2.append(np.concatenate(lst, axis=1))
#         lst2 += (px2[y],)
#     px = np.concatenate(lst2, axis=0)
#     px = np.array(px, 'uint8')
#
#     som_image = Image.fromarray(px)
#     #  som_image.show()
#     return som_image
#
# def display_som_channel(self, som_list, channel):
#     som_list = som_list * 255
#     px2 = []
#     lst2 = ()
#
#     for y in range(neuron_nbr):
#         lst = ()
#         for x in range(neuron_nbr):
#             som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(pictures_dim)
#             lst = lst + (som_list[y * neuron_nbr + x],)
#         px2.append(np.concatenate(lst, axis=1))
#         lst2 += (px2[y],)
#     px = np.concatenate(lst2, axis=0)
#     px = np.array(px, 'uint8')
#
#     som_image = Image.fromarray(px)
#     #  som_image.show()
#     return som_image
