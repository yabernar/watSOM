import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class MosaicImage:
    def __init__(self, image, parameters):
        self.data = None
        self.image = image
        self.pictures_dim = parameters["pictures_dim"]
        self.color = False
        self.nb_pictures = None
        self.clip()

    def clip(self):
        self.data = []
        size = np.flip(self.image.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        print(size)
        pixels = np.array(self.image.getdata(), 'uint8')
        print(pixels.shape)
        if len(pixels.shape) == 2:  # File has RGB colours
            size[1] *= 3
            self.pictures_dim[1] *= 3
            self.color = True
        pixels = pixels.reshape(size)
        self.nb_pictures = np.array(np.divide(size, self.pictures_dim), dtype=int)
        pixels = pixels[0:self.nb_pictures[0] * self.pictures_dim[0],
                 0:self.nb_pictures[1] * self.pictures_dim[1]]  # Cropping the image to make it fit
        px = np.vsplit(pixels, self.nb_pictures[0])
        for i in px:
            j = np.hsplit(i, self.nb_pictures[1])
            for picture in j:
                self.data.append(picture.flatten())
        self.data = np.array(self.data) / 255

    def reconstruct(self, data, size=None):
        if size is None:
            size = self.nb_pictures
        pixels = [None]*size[0]*size[1]
        dim = self.pictures_dim
        if self.color:
            dim = [self.pictures_dim[0], self.pictures_dim[1] // 3, 3]
        px2 = []
        lst2 = ()
        for i in range(size[0]):
            lst = ()
            for j in range(size[1]):
                pixels[i * size[1] + j] = data[i * size[1] + j].reshape(dim)
                lst = lst + (pixels[i * size[1] + j],)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[i],)
        px = np.concatenate(lst2, axis=0)
        px *= 255
        px = np.array(px, 'uint8')
        return px

    def get_data(self):
        return self.data

    def display(self):
        return plt.imshow(self.image)


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
