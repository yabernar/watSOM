from Code.Parameter import Parameter
from Code.Common_Functions import *
from Data.Generated_Data import *


class SOM:
    def __init__(self, inputs):
        # Parameters
        self.alpha = inputs["alpha"]
        self.sigma = inputs["sigma"]
        self.data = inputs["data"]
        self.neurons_nbr = inputs["neurons_nbr"]
        self.epochs_nbr = inputs["epochs_nbr"]

        # Computing variables
        self.neurons = np.zeros(self.neurons_nbr + self.data.shape[1:], dtype=float)
        self.vector_list = None
        self.distance_vector = np.empty(np.sum(self.neurons_nbr))
        self.iteration = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]

    def winner(self, vector):
        dist = np.empty(self.neurons_nbr, dtype=float)
        for i in np.ndindex(dist.shape):
            dist[i] = quadratic_distance(self.neurons[i], vector)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

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
            dist = manhattan_distance(np.asarray(i), np.asarray(bmu))
            self.neurons[i] += self.alpha.get() * self.distance_vector[dist] * (vector - self.neurons[i])

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        return self.vector_list.pop(0)

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)

    def get_neural_map(self):
        return self.neurons

    def get_neural_list(self):
        return self.neurons

    def mean_error(self, winners=None):
        if not winners:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean(np.abs(self.data[i] - self.neurons[winners[i]]))
        return np.mean(error)

    def square_error(self, winners=None):
        if not winners:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.neurons[winners[i]])**2)
        return np.mean(error)


if __name__ == '__main__':
    nb_epochs = 50
    inputs = {"alpha": Parameter(start=0.6, end=0.05, nb_steps=nb_epochs),
              "sigma": Parameter(start=0.5, end=0.001, nb_steps=nb_epochs),
              "data": sierpinski_carpet(200),
              "neurons_nbr": (10, 10, 3),
              "epochs_nbr": nb_epochs}
    som = SOM(inputs)
    som.run()
    print(som.mean_error(som.get_all_winners()))
    print(som.square_error(som.get_all_winners()))
