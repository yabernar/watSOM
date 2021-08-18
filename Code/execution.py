import codecs
#import copy
import os
import json
import numpy as np
from PIL import Image

from Code.FastSOM import FastSOM
from Code.GNG import GrowingNeuralGas
from Code.Parameters import Parameters, Variable
from Code.Recursive_SOM import RecursiveSOM
from Code.SOM import SOM
from Code.cdnet_evaluation import Comparator
from Code.tracking_metrics import TrackingMetrics
from Data.Mosaic_Image import MosaicImage
from Data.random_image import RandomImage


class Execution:
    def __init__(self):
        self.metadata = {}
        self.dataset = {}
        self.model = {}
        self.codebooks = {}
        self.metrics = {}

        self.data = None
        self.training_data = None
        self.map = None

    def open(self, path):
        txt = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(txt)
        self.metadata = data["metadata"]
        self.dataset = data["dataset"]
        self.model = data["model"]
        self.codebooks = data["codebooks"]
        self.metrics = data["metrics"]

    def light_open(self, path):
        txt = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(txt)
        self.metadata = data["metadata"]
        self.dataset = data["dataset"]
        self.model = data["model"]
        self.metrics = data["metrics"]

    def save(self, path):
        data = {"metadata": self.metadata, "dataset": self.dataset, "model": self.model, "metrics": self.metrics,
                "codebooks": self.codebooks}
        json.dump(data, codecs.open(os.path.join(path, self.metadata["name"] + ".json"), 'w', encoding='utf-8'),
                  indent=2)

    def load_dataset(self):
        if self.dataset["type"] == "image":
            path = os.path.join("Data", "images", self.dataset["file"])
            img = Image.open(path)
            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})
            self.data = MosaicImage(img, parameters)
        elif self.dataset["type"] == "random_image":
            path = os.path.join("Data", "images", self.dataset["file"])
            img = Image.open(path)
            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})
            self.data = MosaicImage(img, parameters)
            self.training_data = RandomImage(img, parameters)
        elif self.dataset["type"] == "tracking":
            path = os.path.join("Data", "tracking", "dataset", self.dataset["file"], "bkg.jpg")
            #if self.metadata["seed"] % 2 == 1:
            #    path = os.path.join("Data", "tracking", "dataset", self.dataset["file"], "input", "bkg.jpg")
            #else:
            #    path = os.path.join("Data", "tracking", "dataset", self.dataset["file"], "input", "bkg2.jpg")
            img = Image.open(path)
            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})
            self.data = MosaicImage(img, parameters)
        else:
            print("Error : No dataset type specified !")

    def run(self):
        #if self.metadata["seed"] % 2 == 1:
        np.random.seed(self.metadata["seed"])
        #else:
        #    np.random.seed(self.metadata["seed"]-1)
        if self.model["model"] == "gng":
            self.runGNG()
        else:
            self.runSOM()

    def runGNG(self):
        if self.data is None:
            self.load_dataset()
        inputs = Parameters({"epsilon_winner": 0.1,
                             "epsilon_neighbour": 0.006,
                             "maximum_age": 10,
                             "error_decrease_new_unit": 0.5,
                             "error_decrease_global": 0.995,
                             "data": self.data.get_data(),
                             "neurons_nbr": self.model["nb_neurons"],
                             "epochs_nbr": self.model["nb_epochs"]})
        self.map = GrowingNeuralGas(inputs)
        self.map.run()

    def runSOM(self):
        if self.data is None:
            self.load_dataset()
        nb_epochs = self.model["nb_epochs"]
        if "alpha_start" not in self.model: self.model["alpha_start"] = 0.2
        if "alpha_end" not in self.model: self.model["alpha_end"] = 0.05
        if "sigma_start" not in self.model: self.model["sigma_start"] = 0.7
        if "sigma_end" not in self.model: self.model["sigma_end"] = 0.015
        if "nb_images_evals" not in self.dataset: self.dataset["nb_images_evals"] = 75
        parameters = Parameters({"alpha": Variable(start=self.model["alpha_start"], end=self.model["alpha_end"], nb_steps=nb_epochs),
                                 "sigma": Variable(start=self.model["sigma_start"], end=self.model["sigma_end"], nb_steps=nb_epochs),
                                 "data": self.data.get_data(),
                                 "neurons_nbr": (self.model["width"], self.model["height"]),
                                 "epochs_nbr": nb_epochs})
        if self.model["model"] == "standard":
            self.map = SOM(parameters)
        elif self.model["model"] == "fast":
            self.map = FastSOM(parameters)
        elif self.model["model"] == "recursive":
            self.map = RecursiveSOM(parameters)
        else:
            print("Error : Unknown model !")

        # if "initialisation" not in self.codebooks:
        #     self.codebooks["initialisation"] = self.som.neurons.tolist()
        if "final" in self.codebooks:
            self.map.neurons = np.asarray(self.codebooks["final"])
        else:
            self.map.run()
            #for i in range(nb_epochs):
            #    self.map.run_epoch()
            #self.codebooks["final"] = copy.deepcopy(self.map.neurons.tolist())

        # for i in range(nb_epochs):
        #     print("Epoch "+str(i+1))
        #     if "Epoch "+str(i + 1) not in self.codebooks:
        #         if self.training_data is not None:
        #             self.som.data = self.training_data.get_data(self.som.data.shape[0])
        #         self.som.run_epoch()
        #         # self.codebooks["Epoch " + str(i + 1)] = copy.deepcopy(self.som.neurons.tolist())
        #     self.som.run_epoch()
        self.map.data = self.data.get_data()

    def compute_metrics(self):
        self.metrics["Square_error"] = self.map.square_error()
        # self.metrics["Neurons"] = len(self.som.network.nodes())
        # self.metrics["Connections"] = len(self.som.network.edges())
        if self.dataset["type"] == "tracking":
            current_path = os.path.join("Data", "tracking", "dataset", self.dataset["file"])
            input_path = os.path.join(current_path, "input")
            roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
            temporal_roi = (int(roi_file[0]), int(roi_file[1]))
            mask_roi = Image.open(os.path.join(current_path, "ROI.png"))

            nb_img_gen = self.dataset["nb_images_evals"]
            step = 1
            if nb_img_gen > 0:
                step = (temporal_roi[1] + 1 - temporal_roi[0]) // nb_img_gen

            base = os.path.join("Results", "EpochsNbr2", self.metadata["name"], "results")
            output_path = os.path.join(base, self.dataset["file"])
            supplements_path = os.path.join(base, "supplements")

            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]], "step": step})

            trackingMetric = TrackingMetrics(input_path, output_path, supplements_path, temporal_roi, mask_roi,
                                             parameters=parameters)
            trackingMetric.compute(self.map)
            cmp = Comparator()
            fmeasure, precision, recall = cmp.evaluate__folder_c(current_path, output_path, step)
            # print(fitness)
            self.metrics["fmeasure"] = fmeasure
            self.metrics["precision"] = precision
            self.metrics["recall"] = recall

    def compute_varying_threshold_metric(self):
        if self.dataset["type"] == "tracking":
            current_path = os.path.join("Data", "tracking", "dataset", self.dataset["file"])
            roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
            temporal_roi = (int(roi_file[0]), int(roi_file[1]))

            base = os.path.join("Results", "SOM_Executions", self.metadata["name"], "results")
            output_path = os.path.join(base, self.dataset["file"])
            supplements_path = os.path.join(base, "supplements")
            difference_path = os.path.join(supplements_path, "saliency")

            nb_img_gen = self.dataset["nb_images_evals"]
            step = 1
            if nb_img_gen > 0:
                step = (temporal_roi[1] + 1 - temporal_roi[0]) // nb_img_gen

            ranges = list(range(1, 20)) + list(range(20, 101, 5))
            for threshold in ranges:
                for img in os.listdir(difference_path):
                    som_difference = Image.open(os.path.join(difference_path, img))
                    # Binarizing
                    fn = lambda x: 255 if x > threshold else 0
                    thresholded = som_difference.convert('L').point(fn, mode='1')
                    result = Image.new("L", som_difference.size)
                    result.paste(thresholded, (0, 0))
                    result.save(os.path.join(output_path, "bin"+img[3:]))

                cmp = Comparator()
                fitness = cmp.evaluate__folder_c(current_path, output_path, step)
                # print(fitness)
                self.metrics["fmeasure-t" + str(threshold)] = fitness

    def compute_steps_metrics(self):
        # self.metrics["Square_error"] = self.som.square_error()
        # self.metrics["Neurons"] = len(self.som.network.nodes())
        # self.metrics["Connections"] = len(self.som.network.edges())
        if self.dataset["type"] == "tracking":
            current_path = os.path.join("Data", "tracking", "dataset", self.dataset["file"])
            input_path = os.path.join(current_path, "input")
            roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
            temporal_roi = (int(roi_file[0]), int(roi_file[1]))
            mask_roi = Image.open(os.path.join(current_path, "ROI.png"))

            base = os.path.join("Results", "NbImageEvals", self.metadata["name"], "results")
            output_path = os.path.join(base, self.dataset["file"])
            supplements_path = os.path.join(base, "supplements")

            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})

            # ranges = list(range(1, 5)) + list(range(5, 101, 5))
            #res = []
            #ranges = range(1,201)
            #for i in ranges:
            #    cmp = Comparator()
            #    fitness = cmp.evaluate__folder_c(current_path, output_path, i)
            #    res.append(fitness)
            #self.metrics["fmeasure-steps"] = res

            res = []
            nb_image_ranges = range(5,201)
            for i in nb_image_ranges:
                step = (temporal_roi[1] + 1 - temporal_roi[0]) // i
                cmp = Comparator()
                fitness = cmp.evaluate__folder_c(current_path, output_path, step)
                res.append(fitness)
            self.metrics["fmeasure-nbimgs"] = res

    def full_step_evaluation(self, path):
        self.compute_steps_metrics()
        self.save(path)
        print("Simulation", self.metadata["name"], "ended")

    def full_simulation(self, path):
        if "fmeasure" not in self.metrics:
            self.run()
            self.compute_metrics()
            self.save(path)
        print("Simulation", self.metadata["name"], "ended")

if __name__ == '__main__':
    exec = Execution()
    # exec.metadata = {"name": "test", "seed": 1}
    # exec.dataset = {"type": "random_image", "file": "Lenna.png", "width": 10, "height": 10}
    # exec.model = {"model": "standard", "nb_epochs": 2, "width": 10, "height": 10}
    # exec.run()
    # exec.compute_metrics()
    # exec.save(os.path.join("Executions", "Test"))
    # ex ec.open(os.path.join("Executions", "Test", "test.json"))
    # exec.run()

    # exec = Execution()
    exec.metadata = {"name": "test", "seed": 1}
    exec.dataset = {"type": "tracking", "file": "baseline/highway", "width": 10, "height": 10}
    exec.model = {"model": "standard", "nb_epochs": 2, "width": 10, "height": 10}
    # exec.model = {"model": "gng", "nb_epochs": 20}
    exec.run()
    exec.compute_metrics()
    os.makedirs(os.path.join("Executions", "Test"), exist_ok=True)
    exec.save(os.path.join("Executions", "Test"))
