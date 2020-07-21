import time

from Code.Parameters import Variable, Parameters
from Code.Common_Functions import *
from Data.Generated_Data import *


class SOM:
    def __init__(self, parameters):
        # Parameters
        self.alpha = parameters["alpha"]
        self.sigma = parameters["sigma"]
        self.data = parameters["data"]
        self.neurons_nbr = parameters["neurons_nbr"]
        self.epochs_nbr = parameters["epochs_nbr"]

        # Computing variables
        self.neurons = np.random.random(self.neurons_nbr + self.data.shape[1:])
        self.vector_list = None
        self.distance_to_input = None
        # self.distance_vector = np.empty(np.sum(self.neurons_nbr))
        self.distance_vector = np.empty(manhattan_distance((0, 0), self.neurons_nbr))
        self.iteration = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]

    def set_data(self, input_data):
        self.data = input_data

    def winner(self, vector):
        dist = np.empty(self.neurons_nbr, dtype=float)
        for i in np.ndindex(dist.shape):
            dist[i] = quadratic_distance(self.neurons[i], vector)
#            dist[i] = fast_ied(self.neurons[i], vector)
        self.distance_to_input = dist
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def winner2(self, vector):
        cont = True
        # bmu = (np.random.randint(0, self.neurons_nbr[0]), np.random.randint(0, self.neurons_nbr[1]))
        bmu = (np.int((self.current_vector_index / 62) / 36 * self.neurons_nbr[0]),
              np.int((self.current_vector_index % 62) / 62 * self.neurons_nbr[1]))
        best = quadratic_distance(self.neurons[bmu], vector)
        while cont:
            cont = False
            neighbors = []
            if bmu[0] < self.neurons_nbr[0]-1:
                neighbors.append((bmu[0]+1, bmu[1]))
            if bmu[0] > 0:
                neighbors.append((bmu[0]-1, bmu[1]))
            if bmu[1] < self.neurons_nbr[1]-1:
                neighbors.append((bmu[0], bmu[1]+1))
            if bmu[1] > 0:
                neighbors.append((bmu[0], bmu[1]-1))
            for i in neighbors:
                value = quadratic_distance(self.neurons[i[0], i[1]], vector)
                if value < best:
                    cont = True
                    best = value
                    bmu = i
        return bmu


    def get_all_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.winner(self.data[i])
        return winners_list

    def get_reconstructed_data(self, winners=None):
        if winners is None:
            winners = self.get_all_winners()
        r_data = []
        for index in winners:
            r_data.append(self.neurons[index])
        return r_data

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
            dist = manhattan_distance(np.asarray(i), np.asarray(bmu))
            # dist = hexagonal_distance(np.asarray(i), np.asarray(bmu))
            self.neurons[i] += self.alpha.get() * self.distance_vector[dist] * (vector - self.neurons[i])

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        self.current_vector_index = self.vector_list.pop(0)
        return self.current_vector_index

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)

    def get_neural_map(self):
        return self.neurons

    def get_neural_list(self):
        return np.reshape(self.neurons, (-1,) + self.data.shape[1:])

    def get_neural_distances(self, old_winners, new_winners):
        diff_winners = np.zeros(old_winners.shape)
        for j in range(len(old_winners)):
            diff_winners[j] = manhattan_distance(np.asarray(new_winners[j]), np.asarray(old_winners[j]))
        return diff_winners

    def mean_error(self, winners=None):
        if winners is None:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean(np.abs(self.data[i] - self.neurons[winners[i]]))
        return np.mean(error)

    def square_error(self, winners=None):
        if winners is None:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.neurons[winners[i]])**2)
        return np.mean(error)


if __name__ == '__main__':
    start = time.time()

    nb_epochs = 50
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": sierpinski_carpet(200),
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
    som = SOM(inputs)
    som.run()

    end = time.time()
    print("Executed in " + str(end - start) + " seconds.")
    print(som.mean_error(som.get_all_winners()))
    print(som.square_error(som.get_all_winners()))


