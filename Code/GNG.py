# coding: utf-8
import copy
import os

import numpy as np
from PIL import Image
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt

__authors__ = 'Adrien Guille'
__email__ = 'adrien.guille@univ-lyon2.fr'

from Code.Parameters import Parameters
from Data.Mosaic_Image import MosaicImage

'''
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''


class GrowingNeuralGas:

    def __init__(self, input_data):
        self.network = None
        self.data = copy.deepcopy(input_data)
        self.working_data = copy.deepcopy(input_data)
        self.img = None
        self.units_created = 0
        plt.style.use('ggplot')

    def set_data(self, input_data):
        self.data = copy.deepcopy(input_data)
        self.working_data = copy.deepcopy(input_data)

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        to_be_removed = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                to_be_removed.append((u, v))
        for e in to_be_removed:
            self.network.remove_edge(e[0], e[1])
        to_be_removed = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                to_be_removed.append(u)
        for u in to_be_removed:
            self.network.remove_node(u)

    def fit_network(self, e_b, e_n, a_max, l, a, d, passes=1, plot_evolution=False):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(0, 1) for _ in range(np.shape(self.data)[1])]
        w_b = [np.random.uniform(0, 1) for _ in range(np.shape(self.data)[1])]
        self.network = nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
        # 1. iterate through the data
        sequence = 0
        for p in range(passes):
            # if p % 20 == 0 and p > 0:
            #     self.display()
            # print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # 3. increment the age of all edges emanating from s_1
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. add the squared distance between the observation and the nearest unit in input space
                self.network.nodes[s_1]['error'] += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
                # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
                #    e_b and e_n, respectively, of the total distance
                update_w_s_1 = e_b * (np.subtract(observation, self.network.nodes[s_1]['vector']))
                self.network.nodes[s_1]['vector'] = np.add(self.network.nodes[s_1]['vector'], update_w_s_1)
                update_w_s_n = e_n * (np.subtract(observation, self.network.nodes[s_1]['vector']))
                for neighbor in self.network.neighbors(s_1):
                    self.network.nodes[neighbor]['vector'] = np.add(self.network.nodes[neighbor]['vector'], update_w_s_n)
                # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                self.network.add_edge(s_1, s_2, age=0)
                # 7. remove edges with an age larger than a_max
                #    if this results in units having no emanating edges, remove them as well
                self.prune_connections(a_max)
                # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
                steps += 1
                if steps % l == 0:
                    # if plot_evolution:
                    #     self.plot_network('visualization/sequence/' + str(sequence) + '.png')
                    sequence += 1
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
                    #     remove the original edge between q and f
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    # 8.d decrease the error variables of q and f by multiplying them with a
                    #     initialize the error variable of r with the new value of the error variable of q
                    self.network.nodes[q]['error'] *= a
                    self.network.nodes[f]['error'] *= a
                    self.network.nodes[r]['error'] = self.network.nodes[q]['error']
                # 9. decrease all error variables by multiplying them with a constant d
                error = 0
                for u in self.network.nodes():
                    error += self.network.nodes[u]['error']
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.nodes[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            global_error.append(self.compute_global_error())
        # plt.clf()
        # plt.title('Accumulated local error')
        # plt.xlabel('iterations')
        # plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        # plt.savefig('visualization/accumulated_local_error.png')
        # plt.clf()
        # plt.title('Global error')
        # plt.xlabel('passes')
        # plt.plot(range(len(global_error)), global_error)
        # plt.savefig('visualization/global_error.png')
        # plt.clf()
        # plt.title('Neural network properties')
        # plt.plot(range(len(network_order)), network_order, label='Network order')
        # plt.plot(range(len(network_size)), network_size, label='Network size')
        # plt.legend()
        # plt.savefig('visualization/network_properties.png')

    def winner(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking[0]

    def get_all_winners(self):
        winners_list = np.zeros(self.working_data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.winner(self.working_data[i])
        return winners_list

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
        for i in range(len(winners)):
            # print(i, winners.shape, winners[i], old_winners[i])
            # if winners[i] == old_winners[i]:
            #     distances[i] = 0
            # else:
            try:
                distances[i] = nx.shortest_path_length(self.network, winners[i], old_winners[i])
            except nx.NetworkXNoPath:
                distances[i] = self.units_created
        return distances

    # def plot_network(self, file_path):
    #     plt.clf()
    #     plt.scatter(self.data[:, 0], self.data[:, 1])
    #     node_pos = {}
    #     for u in self.network.nodes():
    #         vector = self.network.nodes[u]['vector']
    #         node_pos[u] = (vector[0], vector[1])
    #     nx.draw(self.network, pos=node_pos)
    #     plt.draw()
    #     plt.savefig(file_path)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    # def plot_clusters(self, clustered_data):
    #     number_of_clusters = nx.number_connected_components(self.network)
    #     plt.clf()
    #     plt.title('Cluster affectation')
    #     color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
    #     for i in range(number_of_clusters):
    #         observations = [observation for observation, s in clustered_data if s == i]
    #         if len(observations) > 0:
    #             observations = np.array(observations)
    #             plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
    #     plt.legend()
    #     plt.savefig('visualization/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
        return global_error

    def square_error(self, winners=None):
        if not winners:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.working_data[i] - self.network.nodes[winners[i]]['vector'])**2)
        return np.mean(error)

    def display(self):
        reconstructed = self.img.reconstruct(self.get_reconstructed_data())
        # som_image = mosaic.reconstruct(gng.get_neural_list(), size=som.neurons_nbr)
        print("Error : ", self.square_error())
        print("Neurons : ", len(self.network.nodes()))
        print("Connections : ", len(self.network.edges()))
        plt.imshow(reconstructed)
        plt.show()


if __name__ == '__main__':
    img = MosaicImage(Image.open(os.path.join("Data", "Tracking", "Dataset", "baseline", "highway", "bkg.jpg")), Parameters({"pictures_dim": [16, 16]}))
    data = img.get_data()
    gng = GrowingNeuralGas(data)
    gng.img = img
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=100, plot_evolution=False)
    reconstructed = img.reconstruct(gng.get_reconstructed_data())
    # som_image = mosaic.reconstruct(gng.get_neural_list(), size=som.neurons_nbr)
    # print(gng.get_all_winners())
    plt.imshow(reconstructed)
    plt.show()
    print(len(gng.network.nodes))

    img2 = MosaicImage(Image.open(os.path.join("Data", "Tracking", "Dataset", "baseline", "highway", "input", "in001010.jpg")), Parameters({"pictures_dim": [16, 16]}))
    data2 = img2.get_data()
    old_winners = gng.get_all_winners()
    gng.data = data2
    gng.working_data = data2
    dists = gng.get_neural_distances(old_winners)
    with np.printoptions(threshold=np.inf, suppress=True, linewidth=720):
        print(dists)
    # plt.imshow(som_image, cmap='gray')
