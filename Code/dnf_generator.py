import itertools
import os
from PIL import Image, ImageChops, ImageOps
import numpy as np
import multiprocessing as mp

from Code.DNF_adrien import DNF
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

        self.tau_dt = parameters["tau_dt"] if parameters["tau_dt"] is not None else 0.8
        self.h = parameters["h"] if parameters["h"] is not None else -0.2
        self.gi = parameters["gi"] if parameters["gi"] is not None else 1

        self.excitation_amplitude = parameters["excitation_amplitude"] if parameters["excitation_amplitude"] is not None else 1
        self.excitation_sigma = parameters["excitation_sigma"] if parameters["excitation_sigma"] is not None else 0.05
        self.inhibition_amplitude = parameters["inhibition_amplitude"] if parameters["inhibition_amplitude"] is not None else 0
        self.inhibition_sigma = parameters["inhibition_sigma"] if parameters["inhibition_sigma"] is not None else 0.5

        self.threshold = parameters["threshold"] if parameters["threshold"] is not None else 10
        self.step = parameters["step"] if parameters["step"] is not None else 1

    def optimize(self):
        os.makedirs(os.path.join(self.supplements_path, "dnf_potentials"), exist_ok=True)

        images_list = sorted([d for d in os.listdir(self.input_path)], key=str.lower)
        first = Image.open(images_list[0])
        start = int(first[8:-4])
        inputs_DNF = {"size": first.size,
                      "tau_dt": self.tau_dt,  # entre 0 et 1
                      "h": self.h,  # entre -1 et 0
                      "Ap": self.excitation_amplitude,  # entre 1 et 10
                      "Sp": self.excitation_sigma,  # entre 0 et 1
                      "Am": self.inhibition_amplitude,  # entre 1 et 10
                      "Sm": self.inhibition_sigma,  # entre 0 et 1
                      "gi": self.gi}  # entre 0 et 10
        dnf = DNF(inputs_DNF)

        for i in range(len(images_list), self.step):
            saliency = Image.open(images_list[i])
            dnf_inputs = np.asarray(saliency)
            dnf.run_once(dnf_inputs)
            dnf_activation = Image.fromarray(dnf.potentials)

            # Binarizing
            fn = lambda x: 255 if x > self.threshold else 0
            thresholded = dnf_activation.convert('L').point(fn, mode='1')

            # Saving
            dnf_activation.save(os.path.join(self.supplements_path, "dnf_activation", "dnf{0:06d}.png".format(start + i)))
            thresholded.save(os.path.join(self.output_path, "bin{0:06d}.png".format(start + i)))
