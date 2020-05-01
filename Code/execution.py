import codecs
import copy
import os
import json
import numpy as np
from PIL import Image

from Code.FastSOM import FastSOM
from Code.Parameters import Parameters, Variable
from Code.SOM import SOM
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
        # self.codebooks["Epoch 30"] = data["codebooks"]["Epoch 30"]
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
        else:
            print("Error : No dataset type specified !")

    def run(self):
        np.random.seed(self.metadata["seed"])
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

    def full_simulation(self, path):
        self.run()
        self.compute_metrics()
        self.save(path)
        print("Simulation", self.metadata["name"], "ended")

if __name__ == '__main__':
    exec = Execution()
    exec.metadata = {"name": "test", "seed": 1}
    exec.dataset = {"type": "random_image", "file": "Lenna.png", "width": 10, "height": 10}
    exec.model = {"model": "standard", "nb_epochs": 2, "width": 10, "height": 10}
    exec.run()
    exec.compute_metrics()
    exec.save(os.path.join("Executions", "Test"))
    # exec.open(os.path.join("Executions", "Test", "test.json"))
    # exec.run()
