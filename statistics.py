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
        self.folder_path = os.path.join("Executions", "Threshold")
        self.results_path = os.path.join("Statistics", "Threshold")

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.light_open(full_path)
                self.all_runs.append(exec)

    def cdnet_surface(self):
        os.makedirs(self.results_path, exist_ok=True)

        videos_files = []
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in categories:
            elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
            for elem in elements:
                videos_files.append(os.path.join(cat, elem))
        for file in videos_files:
            fig = self.surface_graph(file)
            plt.savefig(os.path.join(self.results_path, file.replace("/", "_").replace("\\", "_") + ".png"))

    def surface_graph(self, file=None):
        x = []
        y = []
        z = []
        for e in self.all_runs:
            if file is None or file == e.dataset["file"]:
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

        return fig

    def cdnet_graph(self):
        os.makedirs(self.results_path, exist_ok=True)

        exclusion_list = ["intermittentObjectMotion", "lowFramerate", "PTZ"]

        videos_files = []
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in categories:
            if cat not in exclusion_list:
                elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
                for elem in elements:
                    videos_files.append(os.path.join(cat, elem))
        for file in videos_files:
            fig = self.graph(file)
            plt.savefig(os.path.join(self.results_path, file.replace("/", "_").replace("\\", "_") + ".png"))

    def graph(self, file=None):
        x = range(26)
        y = np.zeros(26)
        w = np.zeros(26)
        z = np.zeros(26)
        for e in self.all_runs:
            if file is None or file == e.dataset["file"]:
                y[e.model["threshold"]] += e.metrics["fmeasure"]
                w[e.model["threshold"]] += e.metrics["precision"]
                z[e.model["threshold"]] += e.metrics["recall"]
        zmin, zmax = 0, 1


        fig = plt.figure()

        # ax.set_zlim(0.95 * zmin, 1.05 * zmax)
        ax = fig.gca()
        ax.set_xlabel("Threshold")
        ax.set_ylabel("fmeasure")
        # plt.xlabel("Taille d'imagettes")
        # ax.invert_xaxis()
        # scatter = ax.scatter(x, y, z, cmap=cm.coolwarm)

        scatter = ax.scatter(x, y, c="b")
        scatter = ax.scatter(x, w, c="r")
        scatter = ax.scatter(x, z, c="g")

        return fig

    def all_graph(self, file=None):
        x = range(26)
        y = np.zeros(26)
        w = np.zeros(26)
        z = np.zeros(26)
        for e in self.all_runs:
            if file is None or file == e.dataset["file"]:
                y[e.model["threshold"]] += e.metrics["fmeasure"]
                w[e.model["threshold"]] += e.metrics["precision"]
                z[e.model["threshold"]] += e.metrics["recall"]
        zmin, zmax = 0.35, 0.4

        y = y / 39
        w = w / 39
        z = z / 39


        fig = plt.figure()

        ax = fig.gca()
        ax.set_ylim(0.95 * zmin, 1.05 * zmax)
        ax.set_xlabel("threshold")
        ax.set_ylabel("fmeasure")
        # plt.xlabel("Taille d'imagettes")
        # ax.invert_xaxis()
        # scatter = ax.scatter(x, y, z, cmap=cm.coolwarm)
        scatter = ax.scatter(x, y, c="b")
        scatter = ax.scatter(x, w, c="r")
        scatter = ax.scatter(x, z, c="g")

        return fig

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
    sr.cdnet_graph()
    # sr.surface_graph()
    # sr.variations_boxplot()
    # sr.all_graph()
    # plt.show()

