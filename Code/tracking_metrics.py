import itertools
import os
from PIL import Image, ImageChops, ImageOps
import numpy as np
import multiprocessing as mp

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM, manhattan_distance
from Data.Mosaic_Image import MosaicImage

np.set_printoptions(threshold=np.inf)


class TrackingMetrics:

    def __init__(self, input_path, output_path, supplements_path, temporal_ROI, mask_ROI, parameters=None):
        self.mask = mask_ROI
        self.input_path = input_path
        self.output_path = output_path
        self.supplements_path = supplements_path
        self.temporal_ROI = temporal_ROI

        if parameters is None:
            parameters = Parameters()
        self.pictures_dim = parameters["pictures_dim"] if parameters["pictures_dim"] is not None else [10, 10]
        self.threshold = parameters["threshold"] if parameters["threshold"] is not None else 10
        self.step = parameters["step"] if parameters["step"] is not None else 1  # IF CHANGED, NEED TO CHANGE COMPARATOR TOO
        self.cut = parameters["cut"] if parameters["cut"] is not None else 0

        self.image_parameters = Parameters({"pictures_dim": self.pictures_dim})

    def compute(self, som):
        os.makedirs(os.path.join(self.supplements_path, "reconstructed"), exist_ok=True)
        os.makedirs(os.path.join(self.supplements_path, "difference"), exist_ok=True)
        os.makedirs(os.path.join(self.supplements_path, "diff_winners"), exist_ok=True)
        os.makedirs(os.path.join(self.supplements_path, "saliency"), exist_ok=True)
        os.makedirs(os.path.join(self.supplements_path, "thresholded"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path), exist_ok=True)
        self.som = som
        self.initial_map = self.som.get_all_winners()

        indexes = range(self.temporal_ROI[0], self.temporal_ROI[1]+1, self.step)
        for i in indexes:
            # print("Extracting ", i)
            self.extract_image(self.input_path, self.output_path, self.supplements_path, i)

    def extract_image(self, input_path, output_path, supplements_path, image_nb, save=True):
        current = Image.open(os.path.join(input_path, "in{0:06d}.jpg".format(image_nb)))
        new_data = MosaicImage(current, self.image_parameters)
        self.som.set_data(new_data.get_data())
        winners = self.som.get_all_winners()
        diff_winners = np.zeros(winners.shape)
        # for j in range(len(winners)):
        #     diff_winners[j] = manhattan_distance(np.asarray(winners[j]), np.asarray(self.initial_map[j]))
        diff_winners = self.som.get_neural_distances(self.initial_map, winners)
        diff_winners -= self.cut
        diff_winners[diff_winners < 0] = 0
        diff_winners[diff_winners > 0] = 1  # New try
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

        # result = ImageChops.multiply(thresholded, self.mask)
        result = Image.new("L", current.size)
        result.paste(thresholded, (0, 0))
        # Saving
        if save:
            reconstructed.save(os.path.join(supplements_path, "reconstructed", "rec{0:06d}.png".format(image_nb)))
            som_difference.save(os.path.join(supplements_path, "difference", "dif{0:06d}.png".format(image_nb)))
            diff_winners.save(os.path.join(supplements_path, "diff_winners", "win{0:06d}.png".format(image_nb)))
            som_difference_modulated.save(os.path.join(supplements_path, "saliency", "sal{0:06d}.png".format(image_nb)))
            thresholded.save(os.path.join(supplements_path, "thresholded", "thr{0:06d}.png".format(image_nb)))

            result.save(os.path.join(output_path, "bin{0:06d}.png".format(image_nb)))