# coding: utf-8
import copy
import os

import numpy as np
from PIL import Image
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt

from Code.Parameters import Parameters
from Data.Mosaic_Image import MosaicImage


class NeuralTopology:
    def __init__(self, input_data):
        self.network = None
        self.data = input_data

        # Memory variables
        self.winners = None

    def set_data(self, input_data):
        self.winners = None
        self.data = input_data

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def get_all_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.find_nearest_units(self.data[i])[0]
        self.winners = winners_list
        return winners_list

    def get_reconstructed_data(self):
        if self.winners is None:
            self.get_all_winners()
        r_data = []
        for u in self.winners:
            r_data.append(self.network.nodes[u]['vector'])
        return r_data

    def get_neural_distances(self, old_winners):
        if self.winners is None:
            self.get_all_winners()
        distances = np.zeros(self.winners.shape)
        for n in self.winners:
            distances = nx.shortest_path_length(self.network, self.network.nodes[n], self.network.nodes[old_winners[n]])
        print(distances)

    def square_error(self):
        if self.winners is None:
            self.get_all_winners()
        error = np.zeros(self.winners.shape)
        for i in np.ndindex(self.winners.shape):
            error[i] = np.mean((self.data[i] - self.network.nodes[self.winners[i]]['vector'])**2)
        return np.mean(error)


if __name__ == '__main__':
    img = MosaicImage(Image.open(os.path.join("Data", "Images", "Lenna.png")), Parameters({"pictures_dim": [10, 10]}))
    data = img.get_data()
    gng = NeuralTopology(data)
    gng.img = img
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=500, plot_evolution=False)
    reconstructed = img.reconstruct(gng.get_reconstructed_data())
    # som_image = mosaic.reconstruct(gng.get_neural_list(), size=som.neurons_nbr)
    print(gng.get_all_winners())
    plt.imshow(reconstructed)
    plt.show()
    # plt.imshow(som_image, cmap='gray')
