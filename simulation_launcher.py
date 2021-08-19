import itertools
import multiprocessing as mp
import numpy as np
import os

from Code.execution import Execution


class SimulationRun:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "Sizing")

    def create(self):
        os.makedirs(self.folder_path, exist_ok=True)

        #exclusion_list = ["intermittentObjectMotion", "lowFramerate", "PTZ", "badWeather", "cameraJitter", "nightVideos", "shadow", "thermal", "dynamicBackground", "shadow", "turbulence"]
        exclusion_list = ["PTZ"]

        videos_files = []
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in categories:
            if cat not in exclusion_list:
                elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
                for elem in elements:
                    videos_files.append(os.path.join(cat, elem))

        #videos_files = [videos_files[0]]

        for v in videos_files:
            for i in range(4, 26, 3):
                for j in range(0, 4):
                    for k in range(5, 36, 5):
                        exec = Execution()
                        exec.metadata = {"name": ""+v.replace("/", "_").replace("\\", "_")+str(i)+"n-"+str(k)+"p-"+str(j+1), "seed": j+1}
                        exec.dataset = {"type": "tracking", "file": v, "nb_images_evals": 105, "width": k, "height": k}
                        exec.model = {"model": "standard", "nb_epochs": 120, "width": i, "height": i}
                        #exec.model = {"model": "standard", "nb_epochs": n, "width": i, "height": i}
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

    def evaluate(self, nb_cores=1):
        pool = mp.Pool(nb_cores)
        pool.starmap(Execution.full_step_evaluation, zip(self.all_runs, itertools.repeat(sr.folder_path)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    sr = SimulationRun()
    #sr.create()
    #sr.save()
    sr.open_folder(sr.folder_path)
    sr.compute(16)
    #sr.evaluate(8)
