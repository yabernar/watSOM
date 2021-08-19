# coding: utf-8
import copy
import os
import time

import numpy as np
from PIL import Image
from scipy import spatial
import networkx as nx

from Code.Parameters import Parameters, Variable
from Data.Mosaic_Image import MosaicImage


class GrowingNeuralGas:
    def __init__(self, parameters):
        # Parameters
        self.epsilon_winner = parameters["epsilon_winner"]
        self.epsilon_neighbour = parameters["epsilon_neighbour"]
        self.maximum_age = parameters["maximum_age"]
        self.error_decrease_new_unit = parameters["error_decrease_new_unit"]
        self.error_decrease_global = parameters["error_decrease_global"]
        self.data = parameters["data"]
        self.neurons_nbr = parameters["neurons_nbr"]
        self.epochs_nbr = parameters["epochs_nbr"]

        # Computing variables
        self.network = nx.Graph()
        self.network.add_node(0, vector=[np.random.uniform(0, 1) for _ in range(np.shape(self.data)[1])], error=0)
        self.network.add_node(1, vector=[np.random.uniform(0, 1) for _ in range(np.shape(self.data)[1])], error=0)
        self.units_created = 2
        self.vector_list = None
        self.iteration = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]
        self.new_unit_step = self.max_iterations // self.neurons_nbr

    def set_data(self, input_data):
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

    def prune_connections(self):
        to_be_removed = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > self.maximum_age:
                to_be_removed.append((u, v))
        for e in to_be_removed:
            self.network.remove_edge(e[0], e[1])
        to_be_removed = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                to_be_removed.append(u)
        for u in to_be_removed:
            self.network.remove_node(u)

    def create_new_units(self):
        if self.iteration % self.new_unit_step == self.new_unit_step//2:
            # 8.a determine the unit q with the maximum accumulated error
            q = 0
            error_max = 0
            for u in self.network.nodes():
                if self.network.nodes[u]['error'] > error_max:
                    error_max = self.network.nodes[u]['error']
                    q = u
            # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
            f = -1
            largest_error = -1
            for u in self.network.neighbors(q):
                if self.network.nodes[u]['error'] > largest_error:
                    largest_error = self.network.nodes[u]['error']
                    f = u
            w_r = 0.5 * (np.add(self.network.nodes[q]['vector'], self.network.nodes[f]['vector']))
            r = self.units_created
            self.units_created += 1
            # 8.c insert edges connecting the new unit r with q and f
            # remove the original edge between q and f
            self.network.add_node(r, vector=w_r, error=0)
            self.network.add_edge(r, q, age=0)
            self.network.add_edge(r, f, age=0)
            self.network.remove_edge(q, f)
            # 8.d decrease the error variables of q and f by multiplying them with a
            # initialize the error variable of r with the new value of the error variable of q
            self.network.nodes[q]['error'] *= self.error_decrease_new_unit
            self.network.nodes[f]['error'] *= self.error_decrease_new_unit
            self.network.nodes[r]['error'] = self.network.nodes[q]['error']


    def run_iteration(self):
        if self.iteration >= self.max_iterations:
            return False

        if self.iteration % self.data.shape[0] == 0:
            self.generate_random_list()

        self.iteration += 1
        observation = self.data[self.unique_random_vector()]

        # 2. find the nearest unit s_1 and the second nearest unit s_2
        nearest_units = self.find_nearest_units(observation)
        s_1 = nearest_units[0]
        s_2 = nearest_units[1]
        # 3. increment the age of all edges emanating from s_1
        for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
            self.network.add_edge(u, v, age=attributes['age'] + 1)
        # 4. add the squared distance between the observation and the nearest unit in input space
        self.network.nodes[s_1]['error'] += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector']) ** 2
        # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions e_b and e_n, respectively, of the total distance
        update_w_s_1 = self.epsilon_winner * (np.subtract(observation, self.network.nodes[s_1]['vector']))
        self.network.nodes[s_1]['vector'] = np.add(self.network.nodes[s_1]['vector'], update_w_s_1)
        update_w_s_n = self.epsilon_neighbour * (np.subtract(observation, self.network.nodes[s_1]['vector']))
        for neighbor in self.network.neighbors(s_1):
            self.network.nodes[neighbor]['vector'] = np.add(self.network.nodes[neighbor]['vector'], update_w_s_n)
        # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
        #    if such an edge doesn't exist, create it
        self.network.add_edge(s_1, s_2, age=0)
        # 7. remove edges with an age larger than a_max
        #    if this results in units having no emanating edges, remove them as well
        self.prune_connections()

        # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
        self.create_new_units()

        # 9. decrease all error variables by multiplying them with a constant d
        for u in self.network.nodes():
            self.network.nodes[u]['error'] *= self.error_decrease_global

    def run_epoch(self):
        for i in range(self.data.shape[0]):
            self.run_iteration()

    def run(self):
        for i in range(self.epochs_nbr):
            self.run_epoch()

    def winner(self, observation):
        return self.find_nearest_units(observation)[0]

    def get_all_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.winner(self.data[i])
        return winners_list

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        self.current_vector_index = self.vector_list.pop(0)
        return self.current_vector_index

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)

    def get_reconstructed_data(self, winners=None):
        if winners is None:
            winners = self.get_all_winners()
        r_data = []
        for u in winners:
            r_data.append(self.network.nodes[u]['vector'])
        return r_data

    def get_neural_distances(self, old_winners):
        winners = self.get_all_winners()
        distances = np.zeros(winners.shape)
        longest_path = 0
        N = list(self.network.nodes())
        for i in range(len(N)):
            for j in range(i+1, len(N)):
                try:
                    l = nx.shortest_path_length(self.network, N[i], N[j])
                    if l > longest_path:
                        longest_path = l
                except nx.NetworkXNoPath:
                    pass
        max = 0
        for i in range(len(winners)):
            try:
                distances[i] = nx.shortest_path_length(self.network, winners[i], old_winners[i])
                if distances[i] > max:
                    max = distances[i]
            except nx.NetworkXNoPath:
                distances[i] = longest_path
        return distances

    # def compute_global_error(self):
    #     global_error = 0
    #     for observation in self.data:
    #         nearest_units = self.find_nearest_units(observation)
    #         s_1 = nearest_units[0]
    #         global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
    #     return global_error

    def square_error(self, winners=None):
        if winners is None:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.network.nodes[winners[i]]['vector'])**2)
        return np.mean(error)


if __name__ == '__main__':
    bkg = MosaicImage(Image.open(os.path.join("Data", "tracking", "dataset", "baseline", "office", "bkg.jpg")), Parameters({"pictures_dim": [16, 16]}))
    image = MosaicImage(Image.open(os.path.join("Data", "tracking", "dataset", "baseline", "office", "input", "in001010.jpg")), Parameters({"pictures_dim": [16, 16]}))

    nb_epochs = 100
    inputs = Parameters({"epsilon_winner": 0.1,
                         "epsilon_neighbour": 0.006,
                         "maximum_age": 10,
                         "error_decrease_new_unit": 0.5,
                         "error_decrease_global": 0.995,
                         "data": bkg.get_data(),
                         "neurons_nbr": 200,
                         "epochs_nbr": nb_epochs})
    gng = GrowingNeuralGas(inputs)

    start = time.time()
    gng.run()
    end = time.time()

    print("Executed in " + str(end - start) + " seconds.")
    print(gng.square_error(gng.get_all_winners()))
    reconstructed = bkg.reconstruct(gng.get_reconstructed_data())
    # plt.imshow(reconstructed)
    print(len(gng.network.nodes()))

    old_winners = gng.get_all_winners()
    gng.set_data(image.get_data())
    dists = gng.get_neural_distances(old_winners)
    with np.printoptions(threshold=np.inf, suppress=True, linewidth=720):
        print(dists)
    # plt.imshow(som_image, cmap='gray')
