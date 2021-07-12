import numpy as np
from scipy import signal, special
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def euclidean_dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def gaussian(distance, sigma):
    return np.exp(-(distance / sigma) ** 2 / 2)


def gaussian_distribution(position, size, sigma):
    result = np.empty(size)
    for i in range(size[0]):
        for j in range(size[1]):
            current = (i / size[0], j / size[1])
            result[i, j] = gaussian(euclidean_dist(position, current), sigma)
    return result


def gaussian_distribution_uniform(position, size, sigma):
    result = gaussian_distribution(position, size, sigma)
    result = np.divide(result, np.sum(result))
    return result


class DNF:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        print("Dnf size : ", width, ";", height)
        self.dt = 0.1
        self.tau = 0.5
        self.cexc = 15
        self.sexc = 0.05
        # self.cinh = 1
        # self.sinh = 0.1
        self.gain = 5
        self.h = -4
        self.gi = 250

        self.input = np.zeros([width, height], dtype=float)
        self.potentials = np.zeros([width, height], dtype=float)
        self.activations = np.zeros([width, height], dtype=float)
        self.lateral = np.zeros([width, height], dtype=float)
        self.kernel = np.zeros([width * 2, height * 2], dtype=float)

        self.kernel = (self.cexc * gaussian_distribution_uniform((0.5, 0.5), (self.width*2, self.height*2), self.sexc)) - (self.gi / (self.width*2 *
                                                                                                                                      self.height*2))
        # (self.cinh * gaussian_distribution_uniform((0.5, 0.5), (self.width*2, self.height*2), self.sinh)) - \

        # for i in range(width*2):
        #     for j in range(height*2):
        #         d = np.sqrt(((i/(width*2)-0.5) ** 2 + ((j/(height*2)-0.5) ** 2))) / np.sqrt(0.5)
        #         self.kernel[i, j] = self.difference_of_gaussian(d)

    def difference_of_gaussian(self, distance):
        return self.cexc * np.exp(-(distance ** 2 / (2 * self.sexc ** 2))) - self.cinh * np.exp(-(distance ** 2 / (2 * self.sinh ** 2)))

    def optimized_DoG(self, x, y):
        return self.kernel[np.abs(x[0] - y[0]), np.abs(x[1] - y[1])]

    def gaussian_activity(self, a, b, sigma):
        for i in range(self.width):
            for j in range(self.height):
                current = (i / self.width, j / self.height)
                self.input[i, j] = gaussian(euclidean_dist(a, current), sigma) + gaussian(euclidean_dist(b, current), sigma)

    def update_neuron(self, x):
        # lateral = 0
        # for i in range(self.width):
        #     for j in range(self.height):
        #         lateral += self.potentials[i, j]*self.optimized_DoG((i, j), x)
        self.potentials[x] += self.dt * (-self.potentials[x] + self.h + self.lateral[x] + self.input[x]*self.gain) / self.tau
        # if self.potentials[x] > 1:
        #     self.potentials[x] = 1
        # elif self.potentials[x] < -1:
        #     self.potentials[x] = -1

    def update_map(self):
        self.activations = special.expit(self.potentials)
        self.lateral = signal.fftconvolve(self.activations, self.kernel, mode='same')
        # self.lateral = np.divide(self.lateral, self.width*self.height)*40*40

        # print(self.lateral)
        neurons_list = list(range(self.width * self.height))
        np.random.shuffle(neurons_list)
        for i in neurons_list:
            self.update_neuron((i % self.width, i // self.width))


def updatefig(*args):
    dnf.update_map()
    im.set_array(dnf.potentials)
    return im,


if __name__ == '__main__':
    fig = plt.figure()
    dnf = DNF(45, 45)
    dnf.gaussian_activity((0.2, 0.2), (0.65, 0.65), 0.1)
    # plt.imshow(dnf.kernel)
    im = plt.imshow(dnf.input, cmap='hot', interpolation='nearest', animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
    plt.show()
