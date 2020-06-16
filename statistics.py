import itertools
import multiprocessing as mp
import numpy as np
import os
from Code.execution import Execution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Statistics:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "office_tracking")

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.light_open(full_path)
                self.all_runs.append(exec)

    def surface_graph(self):
        x = []
        y = []
        z = []
        for e in self.all_runs:
            x.append(e.dataset["width"])
            y.append(e.model["width"])
            z.append(e.metrics["fmeasure"])
            # y.append(10 * np.log10(1 / e.metrics["Square_error"]))
        zmin, zmax = min(z), max(z)

        surface = np.zeros((18, 26))
        for e in self.all_runs:
            surface[e.model["width"]-3, e.dataset["width"]-5] += e.metrics["fmeasure"]
        surface = surface/3

        X, Y = np.meshgrid(range(5, 31), range(3, 21))

        # zmin, zmax = min(z), max(z)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_zlim(0.95 * zmin, 1.05 * zmax)
        ax.set_xlabel("Taille imagettes")
        ax.set_ylabel("Taille carte")
        # plt.xlabel("Taille d'imagettes")
        ax.invert_xaxis()
        surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm)

        plt.show()

    def graph(self):
        x = []
        y = []
        z = []
        for e in self.all_runs:
            x.append(e.metrics["Neurons"])
            # y.append(e.metrics["fmeasure"])
            z.append(e.metrics["fmeasure"])
            # y.append(10 * np.log10(1 / e.metrics["Square_error"]))
        zmin, zmax = min(z), max(z)

        fig = plt.figure()
        # ax = fig.gca(projection='3d')

        # ax.set_zlim(0.95 * zmin, 1.05 * zmax)
        plt.xlabel("Nb Neurons")
        plt.ylabel("fmeasure")
        # plt.xlabel("Taille d'imagettes")
        # ax.invert_xaxis()
        # scatter = ax.scatter(x, y, z, cmap=cm.coolwarm)
        plt.scatter(x, z)

        plt.show()


        plt.show()

    def variations_boxplot(self):
        ranges = list(range(1, 5)) + list(range(5, 101, 5))
        data = np.zeros((len(self.all_runs), len(ranges)))
        j = 0
        for e in self.all_runs:
            for i in range(len(ranges)):
                data[j, i] = e.metrics["fmeasure-s"+str(ranges[i])]
            j += 1

        print(data)

        for i in range(len(self.all_runs)):
            data[i, ::] -= data[i, 0]

        print(data)

        fig = plt.figure()
        # ax = fig.gca(projection='3d')

        # ax.set_zlim(0.95 * zmin, 1.05 * zmax)
        plt.xticks([], ranges)
        plt.xlabel("Image generation step")
        plt.ylabel("fmeasure")
        # plt.xlabel("Taille d'imagettes")
        # ax.invert_xaxis()
        # scatter = ax.scatter(x, y, z, cmap=cm.coolwarm)
        plt.boxplot(data)

        plt.show()

if __name__ == '__main__':
    sr = Statistics()
    sr.open_folder(sr.folder_path)
    # sr.graph()
    # sr.surface_graph()
    sr.variations_boxplot()