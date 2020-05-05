import os
import time

import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt

from Code.NeuralTopology import NeuralTopology
from Code.Parameters import Variable, Parameters
from Code.Common_Functions import *
from Data.Generated_Data import *
from Data.Mosaic_Image import MosaicImage


class SelfOrganizingTopology(NeuralTopology):
    def __init__(self, parameters):
        # Parameters
        super().__init__(parameters["data"])
        self.alpha = parameters["alpha"]
        self.sigma = parameters["sigma"]
        self.neurons_nbr = parameters["neurons_nbr"]
        self.epochs_nbr = parameters["epochs_nbr"]

        self.network = nx.Graph()
        for i in range(self.neurons_nbr[0]):
            for j in range(self.neurons_nbr[1]):
                current = i+self.neurons_nbr[0]*j
                self.network.add_node(current, vector=[np.random.uniform(0, 1) for _ in range(np.shape(self.data)[1])])
                if i > 0:
                    self.network.add_edge(current, current-1)
                if j > 0:
                    self.network.add_edge(current, current-self.neurons_nbr[0])

        # Computing variables
        self.vector_list = None
        # self.distance_to_input = None
        self.distance_vector = np.empty(np.sum(self.neurons_nbr))
        # self.distance_vector = np.empty(manhattan_distance((0, 0), self.neurons_nbr))
        self.iteration = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]

    def run_iteration(self):
        if self.iteration >= self.max_iterations:
            return False

        if self.iteration % self.data.shape[0] == 0:
            self.generate_random_list()
            self.alpha.execute()
            self.sigma.execute()
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = normalized_gaussian(i / (len(self.distance_vector) - 1), self.sigma.get())

        self.iteration += 1
        vector = self.data[self.unique_random_vector()]
        bmu = self.find_nearest_units(vector)[0]
        self.updating_weights(bmu, vector)
        return bmu

    def run_epoch(self):
        for i in range(self.data.shape[0]):
            self.run_iteration()

    def run(self):
        for i in range(self.epochs_nbr):
            self.run_epoch()

    def updating_weights(self, bmu, vector):
        for u, attributes in self.network.nodes(data=True):
            dists = nx.shortest_path_length(self.network, source=bmu)
            # dist = hexagonal_distance(np.asarray(i), np.asarray(bmu))
            self.network.nodes[u]['vector'] += self.alpha.get() * self.distance_vector[dists[u]] * (vector - self.network.nodes[i]['vector'])

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        self.current_vector_index = self.vector_list.pop(0)
        return self.current_vector_index

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)


if __name__ == '__main__':
    img = MosaicImage(Image.open(os.path.join("Data", "Images", "Lenna.png")), Parameters({"pictures_dim": [20, 20]}))
    data = img.get_data()

    start = time.time()

    nb_epochs = 20
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": img.get_data(),
                         "neurons_nbr": (5, 5),
                         "epochs_nbr": nb_epochs})
    som = SelfOrganizingTopology(inputs)
    som.run()

    end = time.time()
    print("Executed in " + str(end - start) + " seconds.")
    print(som.square_error())
    reconstructed = img.reconstruct(som.get_reconstructed_data())
    plt.imshow(reconstructed)
    plt.show()



