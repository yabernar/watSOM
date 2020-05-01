import itertools
import multiprocessing as mp
import numpy as np
import os
from Code.execution import Execution


class SimulationRun:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "Epochs")

    def create(self):
        for i in range(102, 201, 2):
            for j in range(3):
                exec = Execution()
                exec.metadata = {"name": "Epoch"+str(i)+"-"+str(j+1), "seed": j+1}
                exec.dataset = {"type": "image", "file": "Lenna.png", "width": 10, "height": 10}
                exec.model = {"model": "standard", "nb_epochs": i, "width": 10, "height": 10}
                self.all_runs.append(exec)

    def save(self):
        for e in self.all_runs:
            e.save(self.folder_path)

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.open(full_path)
                self.all_runs.append(exec)

    def compute(self, nb_cores=1):
        pool = mp.Pool(nb_cores)
        pool.starmap(Execution.full_simulation, zip(self.all_runs, itertools.repeat(sr.folder_path)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    sr = SimulationRun()
    # sr.create()
    # sr.save()
    sr.open_folder(sr.folder_path)
    sr.compute(15)
