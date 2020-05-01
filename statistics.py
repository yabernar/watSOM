import itertools
import multiprocessing as mp
import numpy as np
import os
from Code.execution import Execution
import matplotlib.pyplot as plt


class Statistics:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "MapSize")

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.light_open(full_path)
                self.all_runs.append(exec)

    def graph(self):
        x = []
        y = []
        for e in self.all_runs:
            x.append(e.model["width"])
            y.append(e.metrics["Square_error"])
            # y.append(10 * np.log10(1 / e.metrics["Square_error"]))
        ymin, ymax = min(y), max(y)
        plt.ylim(0.95 * ymin, 1.05 * ymax)
        plt.xlabel("Taille de la carte")
        # plt.xlabel("Taille d'imagettes")
        plt.scatter(x, y)
        plt.show()

if __name__ == '__main__':
    sr = Statistics()
    # sr.create()
    # sr.save()
    sr.open_folder(sr.folder_path)
    # sr.compute(3)
    sr.graph()