import itertools
import os
from PIL import Image, ImageChops, ImageOps
import numpy as np
import multiprocessing as mp

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM, manhattan_distance
from Data.Mosaic_Image import MosaicImage


np.set_printoptions(threshold=np.inf)


class SaliencyGenerator:

    def __init__(self, input_path, output_path, supplements_path, temporal_ROI):
        bkg = Image.open(os.path.join(input_path, "bkg.jpg"))
        self.learning(bkg)
        self.generate(input_path, output_path, supplements_path, temporal_ROI)
        print("--Finished]", flush=True)

    def learning(self, bkg_image):
        print("[Learning--", end='', flush=True)
        # PARAMETERS
        self.pictures_dim = [10, 10]
        nb_epochs = 50
        self.image_parameters = Parameters({"pictures_dim": self.pictures_dim})
        data = MosaicImage(bkg_image, self.image_parameters)
        inputs_SOM = Parameters({"alpha": Variable(start=0.5, end=0.25, nb_steps=nb_epochs),
                                 "sigma": Variable(start=0.1, end=0.03, nb_steps=nb_epochs),
                                 "data": data.get_data(),
                                 "neurons_nbr": (10, 10),
                                 "epochs_nbr": nb_epochs})
        self.som = SOM(inputs_SOM)

        # RUN
        for i in range(nb_epochs):
            # print('Epoch', i)
            self.som.run_epoch()
        self.initial_map = self.som.get_all_winners()

    def generate(self, input_path, output_path, supplements_path, temporal_ROI):
        print("--Tracking--", end = '', flush=True)
        os.makedirs(os.path.join(supplements_path, "difference"), exist_ok=True)
        os.makedirs(os.path.join(supplements_path, "diff_winners"), exist_ok=True)
        os.makedirs(os.path.join(supplements_path, "saliency"), exist_ok=True)

        indexes = range(temporal_ROI[0], temporal_ROI[1]+1)
        nb_runs = len(indexes)
        pool = mp.Pool(8)
        a = pool.starmap(self.extract_image, zip(itertools.repeat(input_path, nb_runs), itertools.repeat(output_path, nb_runs),
                                                 itertools.repeat(supplements_path, nb_runs), indexes))
        pool.close()
        pool.join()

    def extract_image(self, input_path, output_path, supplements_path, image_nb):
        current = Image.open(os.path.join(input_path, "in{0:06d}.jpg".format(image_nb)))
        new_data = MosaicImage(current, self.image_parameters)
        self.som.set_data(new_data.get_data())
        winners = self.som.get_all_winners()
        diff_winners = np.zeros(winners.shape)
        for j in range(len(winners)):
            diff_winners[j] = manhattan_distance(np.asarray(winners[j]), np.asarray(self.initial_map[j]))
        diff_winners = diff_winners.reshape(new_data.nb_pictures)
        diff_winners = np.kron(diff_winners, np.ones((self.pictures_dim[0], self.pictures_dim[1])))
        #             diff_winners *= 30  # Use this parameter ?
        diff_winners = ImageOps.autocontrast(Image.fromarray(diff_winners).convert('L'))

        reconstructed = Image.fromarray(new_data.reconstruct(self.som.get_reconstructed_data(winners)))
        som_difference = ImageOps.autocontrast(ImageChops.difference(reconstructed, current).convert('L'))
        som_difference_modulated = ImageChops.multiply(som_difference, diff_winners)

        # Binarizing
        thresh = 10
        fn = lambda x: 255 if x > thresh else 0
        result = som_difference_modulated.convert('L').point(fn, mode='1')

        # Saving
        som_difference.save(os.path.join(supplements_path, "difference", "dif{0:06d}.png".format(image_nb)))
        diff_winners.save(os.path.join(supplements_path, "diff_winners", "win{0:06d}.png".format(image_nb)))
        som_difference_modulated.save(os.path.join(supplements_path, "saliency", "sal{0:06d}.png".format(image_nb)))
        result.save(os.path.join(output_path, "bin{0:06d}.png".format(image_nb)))


if __name__ == '__main__':
    cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset/baseline/pedestrians/input/"
    supplements_path = "/users/yabernar/GrosDisque/CDNET14/testing"
    os.makedirs(os.path.join(supplements_path, "results"), exist_ok=True)
    output_path = "/users/yabernar/GrosDisque/CDNET14/testing/results"
    roi = (300, 330)
    SaliencyGenerator(cdnet_path, output_path, supplements_path, roi)
