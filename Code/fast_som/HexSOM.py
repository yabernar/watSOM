import time

from Code.Parameters import Variable, Parameters
from Code.Common_Functions import *
from Data.Generated_Data import *


class HexSOM:
    def __init__(self, parameters):
        # Parameters
        self.alpha = parameters["alpha"]
        self.sigma = parameters["sigma"]
        self.data = parameters["data"]
        self.neurons_nbr = parameters["neurons_nbr"]
        self.epochs_nbr = parameters["epochs_nbr"]

        # Computing variables
        self.neurons = np.zeros(self.neurons_nbr + self.data.shape[1:], dtype=float)
        self.vector_list = None
        self.distance_to_input = None
        self.distance_vector = np.empty(hexagonal_distance((0, 0), self.neurons_nbr))
        self.iteration = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]

    def winner(self, vector):
        dist = np.empty(self.neurons_nbr, dtype=float)
        for i in np.ndindex(dist.shape):
            dist[i] = quadratic_distance(self.neurons[i], vector)
#            dist[i] = fast_ied(self.neurons[i], vector)
        self.distance_to_input = dist
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def fast_winner(self, vector):
        self.dist_memory = np.zeros(self.neurons_nbr, dtype=float)
        starting_positions = [(0, 0), (self.neurons_nbr[0]-1, 0), (0, self.neurons_nbr[1]-1), (self.neurons_nbr[0]-1, self.neurons_nbr[1]-1)]
        best = np.inf
        bmu = (0, 0)
        for i in starting_positions:
            current = self.particle(vector, i)
            value = quadratic_distance(self.neurons[current], vector)
            if value < best:
                best = value
                bmu = current
        return bmu

    def particle(self, vector, start):
        cont = True
        bmu = start
        best = quadratic_distance(self.neurons[bmu], vector)
        self.dist_memory[bmu] = best
        while cont:
            new_bmu, new_best = self.search_smallest_neighbor(vector, bmu, best)
            if new_bmu == bmu:
                cont = False
                neighbors = self.return_all_neighbors(bmu)
                for i in neighbors:
                    neighbors_bmu, neighbors_best = self.search_smallest_neighbor(vector, i, np.inf)
                    if neighbors_bmu != bmu:
                        best = neighbors_best
                        bmu = neighbors_bmu
                        cont = True
                        break
            else:
                best = new_best
                bmu = new_bmu
        return bmu

    def return_all_neighbors(self, bmu):
        neighbors = []
        if bmu[0] < self.neurons_nbr[0] - 1:
            neighbors.append((bmu[0] + 1, bmu[1]))
        if bmu[0] > 0:
            neighbors.append((bmu[0] - 1, bmu[1]))
        if bmu[1] < self.neurons_nbr[1] - 1:
            neighbors.append((bmu[0], bmu[1] + 1))
        if bmu[1] > 0:
            neighbors.append((bmu[0], bmu[1] - 1))
        if bmu[1] % 2 == 0 and bmu[0] > 0:
            if bmu[1] > 0:
                neighbors.append((bmu[0] - 1, bmu[1] - 1))
            if bmu[1] < self.neurons_nbr[1] - 1:
                neighbors.append((bmu[0] - 1, bmu[1] + 1))
        if bmu[1] % 2 == 1 and bmu[0] < self.neurons_nbr[0]-1:
            if bmu[1] > 0:
                neighbors.append((bmu[0] + 1, bmu[1] - 1))
            if bmu[1] < self.neurons_nbr[1] - 1:
                neighbors.append((bmu[0] + 1, bmu[1] + 1))
        return neighbors

    def search_smallest_neighbor(self, vector, bmu, best):
        neighbors = self.return_all_neighbors(bmu)
        for i in neighbors:
            value = quadratic_distance(self.neurons[i[0], i[1]], vector)
            self.dist_memory[i] = value
            if value < best:
                best = value
                bmu = i
        return bmu, best

    def get_all_fast_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        error = 0
        for i in np.ndindex(winners_list.shape):
            win = self.winner(self.data[i])
            fast = self.fast_winner(self.data[i])
            if fast != win:
                error += 1
            winners_list[i] = fast
        # print("Error number :", error)
        return winners_list, error

    def get_all_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.winner(self.data[i])
        return winners_list

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
        bmu = self.winner(vector)
        self.updating_weights(bmu, vector)
        return bmu[0], bmu[1]

    def run_epoch(self):
        for i in range(self.data.shape[0]):
            self.run_iteration()

    def run(self):
        for i in range(self.epochs_nbr):
            self.run_epoch()

    def updating_weights(self, bmu, vector):
        for i in np.ndindex(self.neurons_nbr):
            dist = hexagonal_distance(np.asarray(i), np.asarray(bmu))
            self.neurons[i] += self.alpha.get() * self.distance_vector[dist] * (vector - self.neurons[i])

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        self.current_vector_index = self.vector_list.pop(0)
        return self.current_vector_index

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)

    def mean_square_quantization_error(self, winners=None):
        if not winners:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.neurons[winners[i]])**2)
        return np.mean(error)

    def mean_square_quantization_error_secondary(self):
        winners, nb_error = self.get_all_fast_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.neurons[winners[i]])**2)
        return np.mean(error), nb_error

    def mean_square_distance_to_neighbour(self):
        error = np.zeros(self.neurons.shape)
        for i in np.ndindex(self.neurons_nbr):
            nb_neighbours = 0
            for j in np.ndindex(self.neurons_nbr):
                if hexagonal_distance(np.asarray(i), np.asarray(j)) <= 1:
                    error[i] += np.mean((self.neurons[i] - self.neurons[j])**2)
                    nb_neighbours += 1
            error[i] /= nb_neighbours
        return np.mean(error)


if __name__ == '__main__':
    start = time.time()

    nb_epochs = 50
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.2, nb_steps=nb_epochs),
                         "data": uniform(200, 3),
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
    som = HexSOM(inputs)
    som.run()

    end = time.time()
    print("Executed in " + str(end - start) + " seconds.")
    print(som.mean_square_quantization_error())
    print(som.mean_square_distance_to_neighbour())
