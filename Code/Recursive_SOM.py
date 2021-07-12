import time

from PIL import Image

from Code.Parameters import Variable, Parameters
from Code.Common_Functions import *
from Code.SOM import SOM
from Data.Generated_Data import *
from Data.Mosaic_Image import MosaicImage


class RecursiveSOM:
    def __init__(self, parameters):
        self.data = parameters["data"]
        self.level_one = SOM(parameters)
        self.level_two = SOM(parameters)

    def set_data(self, input_data):
        self.data = input_data
        self.level_one.set_data(self.data)
        diff = self.level_one.get_difference()
        diff = (np.asarray(diff) + 1) / 2
        self.level_two.set_data(diff)

    def run(self):
        self.level_one.run()
        diff = self.level_one.get_difference()
        diff = (np.asarray(diff) + 1) / 2
        self.level_two.set_data(diff)
        self.level_two.run()

    def get_reconstructed_data(self, winners=None):
        rec_one = np.asarray(self.level_one.get_reconstructed_data(winners))
        rec_two = np.asarray(self.level_two.get_reconstructed_data())
        rec = rec_one + (rec_two*2) - 1
        for i in rec:
            for j in np.ndindex(i.shape):
                if i[j] < 0: i[j] = 0
                elif i[j] > 1: i[j] = 1
        return rec

    def get_all_winners(self):
        return self.level_one.get_all_winners()

    def get_neural_distances(self, old_winners, new_winners):
        return self.level_one.get_neural_distances(old_winners, new_winners)

    def mean_error(self):
        rec = self.get_reconstructed_data()
        diff = rec - self.data
        error = []
        for i in diff:
            error.append(np.mean(np.abs(i)))
        return np.mean(error)

    def square_error(self):
        rec = self.get_reconstructed_data()
        diff = rec - self.data
        error = []
        for i in diff:
            error.append(np.mean(i**2))
        return np.mean(error)

if __name__ == '__main__':
    start = time.time()
    img = MosaicImage(Image.open("Data/images/Elijah.png"), Parameters({"pictures_dim": [10, 10]}))

    nb_epochs = 100
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": img.get_data(),
                         "neurons_nbr": (8, 8),
                         "epochs_nbr": nb_epochs})
    som = RecursiveSOM(inputs)
    som.run()

    end = time.time()
    print("Executed in " + str(end - start) + " seconds.")
    print("Mean :", som.mean_error(), "compared to", som.level_one.mean_error())
    print("Square : ", som.square_error(), "compared to", som.level_one.square_error())

    two = img.reconstruct(som.get_reconstructed_data())
    one = img.reconstruct(som.level_one.get_reconstructed_data())

    ah = MosaicImage(Image.open("Data/images/ah.png"), Parameters({"pictures_dim": [10, 10]}))
    som.set_data(ah.get_data())

    ah_two = img.reconstruct(som.get_reconstructed_data())
    ah_one = img.reconstruct(som.level_one.get_reconstructed_data())


    first_stage = img.reconstruct(som.level_one.get_neural_list(), size=som.level_one.neurons_nbr)
    second_stage = img.reconstruct(som.level_two.get_neural_list(), size=som.level_two.neurons_nbr)

    Image.fromarray(one).save("Results/Recursive/one.png")
    Image.fromarray(two).save("Results/Recursive/two.png")

    Image.fromarray(ah_one).save("Results/Recursive/ah_one.png")
    Image.fromarray(ah_two).save("Results/Recursive/ah_two.png")

    Image.fromarray(first_stage).save("Results/Recursive/first_stage.png")
    Image.fromarray(second_stage).save("Results/Recursive/second_stage.png")
