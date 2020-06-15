import codecs
import copy
import os
import json
import numpy as np
from PIL import Image

from Code.FastSOM import FastSOM
from Code.GNG import GrowingNeuralGas
from Code.Parameters import Parameters, Variable
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
        self.som = None

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
        data = {"metadata": self.metadata, "dataset": self.dataset, "model": self.model, "metrics": self.metrics, "codebooks": self.codebooks}
        json.dump(data,  codecs.open(os.path.join(path, self.metadata["name"] + ".json"), 'w', encoding='utf-8'),  indent=2)

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
            path = os.path.join("Data", "tracking", "dataset", "baseline", self.dataset["file"], "bkg.jpg")
            img = Image.open(path)
            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})
            self.data = MosaicImage(img, parameters)
        else:
            print("Error : No dataset type specified !")

    def run(self):
        np.random.seed(self.metadata["seed"])
        if self.model["model"] == "gng":
            self.runGNG()
        else:
            self.runSOMS()

    def runGNG(self):
        if self.data is None:
            self.load_dataset()
        nb_epochs = self.model["nb_epochs"]
        self.som = GrowingNeuralGas(self.data.get_data())
        self.som.img = self.data
        self.som.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=nb_epochs, plot_evolution=False)

    def runSOMS(self):
        if self.data is None:
            self.load_dataset()
        nb_epochs = self.model["nb_epochs"]
        parameters = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                                 "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                                 "data": self.data.get_data(),
                                 "neurons_nbr": (self.model["width"], self.model["height"]),
                                 "epochs_nbr": nb_epochs})
        if self.model["model"] == "standard":
            self.som = SOM(parameters)
        elif self.model["model"] == "fast":
            self.som = FastSOM(parameters)
        else:
            print("Error : Unknown model !")

        # if "initialisation" not in self.codebooks:
        #     self.codebooks["initialisation"] = self.som.neurons.tolist()
        if "final" in self.codebooks:
            self.som.neurons = np.asarray(self.codebooks["final"])
        else:
            for i in range(nb_epochs):
                self.som.run_epoch()
            self.codebooks["final"] = copy.deepcopy(self.som.neurons.tolist())

        # for i in range(nb_epochs):
        #     print("Epoch "+str(i+1))
        #     if "Epoch "+str(i + 1) not in self.codebooks:
        #         if self.training_data is not None:
        #             self.som.data = self.training_data.get_data(self.som.data.shape[0])
        #         self.som.run_epoch()
        #         # self.codebooks["Epoch " + str(i + 1)] = copy.deepcopy(self.som.neurons.tolist())
        #     self.som.run_epoch()
        self.som.data = self.data.get_data()

    def compute_metrics(self):
        self.metrics["Square_error"] = self.som.square_error()
        # self.metrics["Neurons"] = len(self.som.network.nodes())
        # self.metrics["Connections"] = len(self.som.network.edges())
        if self.dataset["type"] == "tracking":
            current_path = os.path.join("Data", "tracking", "dataset", "baseline", self.dataset["file"])
            input_path = os.path.join(current_path, "input")
            roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
            temporal_roi = (int(roi_file[0]), int(roi_file[1]))
            mask_roi = Image.open(os.path.join(current_path, "ROI.png"))

            base = os.path.join("Results", "SOM_Executions", self.metadata["name"], "results")
            output_path = os.path.join(base, "baseline", self.dataset["file"])
            supplements_path = os.path.join(base, "supplements")

            parameters = Parameters({"pictures_dim": [self.dataset["width"], self.dataset["height"]]})

            trackingMetric = TrackingMetrics(input_path, output_path, supplements_path, temporal_roi, mask_roi, parameters=parameters)
            trackingMetric.compute(self.som)
            cmp = Comparator()
            fitness = cmp.evaluate__folder_c(current_path, output_path)
            # print(fitness)
            self.metrics["fmeasure"] = fitness

    def full_simulation(self, path):
        self.run()
        self.compute_metrics()
        self.save(path)
        print("Simulation", self.metadata["name"], "ended")

if __name__ == '__main__':
    exec = Execution()
    # exec.metadata = {"name": "test", "seed": 1}
    exec.dataset = {"type": "random_image", "file": "Lenna.png", "width": 10, "height": 10}
    # exec.model = {"model": "standard", "nb_epochs": 2, "width": 10, "height": 10}
    # exec.run()
    # exec.compute_metrics()
    # exec.save(os.path.join("Executions", "Test"))
    # exec.open(os.path.join("Executions", "Test", "test.json"))
    # exec.run()

    # exec = Execution()
    exec.metadata = {"name": "test", "seed": 1}
    # exec.dataset = {"type": "tracking", "file": "highway", "width": 10, "height": 10}
    exec.model = {"model": "standard", "nb_epochs": 2, "width": 10, "height": 10}
    # exec.model = {"model": "gng", "nb_epochs": 20}
    exec.run()
    exec.compute_metrics()
    os.makedirs(os.path.join("Executions", "Test"), exist_ok=True)
    exec.save(os.path.join("Executions", "Test"))
