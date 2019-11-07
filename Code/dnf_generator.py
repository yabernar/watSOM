import itertools
import os
from PIL import Image, ImageChops, ImageOps
import numpy as np
import multiprocessing as mp

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM, manhattan_distance
from Data.Mosaic_Image import MosaicImage

np.set_printoptions(threshold=np.inf)


class DNFGenerator:

    def __init__(self, input_path, output_path, supplements_path, parameters=None):
        self.input_path = input_path
        self.output_path = output_path
        self.supplements_path = supplements_path

        if parameters is None:
            parameters = Parameters()

        inputs_DNF = {"size": (240, 320),
                      "tau_dt": 0.2,  # entre 0 et 1
                      "h": -0.1,  # entre -1 et 0
                      "Ap": 5,  # entre 1 et 10
                      "Sp": 0.06,  # entre 0 et 1
                      "Am": 0,  # entre 1 et 10
                      "Sm": 1,  # entre 0 et 1
                      "gi": 1}  # entre 0 et 10

        self.tau_dt = parameters["tau_dt"] if parameters["tau_dt"] is not None else 0.8
        self.h = parameters["h"] if parameters["h"] is not None else -0.2
        self.gi = parameters["gi"] if parameters["gi"] is not None else 1

        self.alpha_start = parameters["excitation_amplitude"] if parameters["excitation_amplitude"] is not None else 1
        self.alpha_end = parameters["excitation_sigma"] if parameters["excitation_sigma"] is not None else 0.05
        self.sigma_start = parameters["inhibition_amplitude"] if parameters["inhibition_amplitude"] is not None else 0
        self.sigma_end = parameters["inhibition_sigma"] if parameters["inhibition_sigma"] is not None else 0.5

        self.threshold = parameters["threshold"] if parameters["threshold"] is not None else 10
        self.step = parameters["step"] if parameters["step"] is not None else 1

    def generate_all(self):
        print("[Tracking--", end = '', flush=True)
        self.generate(self.input_path, self.output_path, self.supplements_path)
        print("--Finished]", flush=True)

    def optimize(self):
        os.makedirs(os.path.join(self.supplements_path, "dnf_potentials"), exist_ok=True)



        indexes = range(self.temporal_ROI[0], self.temporal_ROI[1]+1, self.step)
        nb_runs = len(indexes)
        for i in indexes:
            self.extract_image(self.input_path, self.output_path, self.supplements_path, i)

    def generate(self, input_path, output_path, supplements_path, temporal_ROI):
        os.makedirs(os.path.join(supplements_path, "difference"), exist_ok=True)
        os.makedirs(os.path.join(supplements_path, "diff_winners"), exist_ok=True)
        os.makedirs(os.path.join(supplements_path, "saliency"), exist_ok=True)
        os.makedirs(os.path.join(supplements_path, "thresholded"), exist_ok=True)


        indexes = range(temporal_ROI[0], temporal_ROI[1]+1)
        nb_runs = len(indexes)
        pool = mp.Pool(8)
        a = pool.starmap(self.extract_image, zip(itertools.repeat(input_path, nb_runs), itertools.repeat(output_path, nb_runs),
                                                 itertools.repeat(supplements_path, nb_runs), indexes, itertools.repeat(True, nb_runs)))
        pool.close()
        pool.join()

    def extract_image(self, input_path, output_path, supplements_path, image_nb, save=True):
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
        fn = lambda x: 255 if x > self.threshold else 0
        thresholded = som_difference_modulated.convert('L').point(fn, mode='1')

        result = ImageChops.multiply(thresholded, self.mask)

        # Saving
        if save:
            som_difference.save(os.path.join(supplements_path, "difference", "dif{0:06d}.png".format(image_nb)))
            diff_winners.save(os.path.join(supplements_path, "diff_winners", "win{0:06d}.png".format(image_nb)))
            som_difference_modulated.save(os.path.join(supplements_path, "saliency", "sal{0:06d}.png".format(image_nb)))
            thresholded.save(os.path.join(supplements_path, "thresholded", "thr{0:06d}.png".format(image_nb)))

            result.save(os.path.join(output_path, "bin{0:06d}.png".format(image_nb)))
