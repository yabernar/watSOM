import itertools
import multiprocessing as mp
import numpy as np
import os

from scipy.stats import stats

from Code.execution import Execution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import pandas as pd

font = {'family' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

class Statistics:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "NbImageEvals")
        self.results_path = os.path.join("Statistics", "Stats")

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

    def generate_csv(self):
        df = pd.DataFrame(columns=["model", "video", "square_error", "fmeasure", "precision", "rappel", "PSNR"])
        for e in self.all_runs:
            df = df.append({"model": e.model["model"],
                       "video": e.dataset["file"][9:],
                       "square_error": e.metrics["Square_error"],
                       "fmeasure": e.metrics["fmeasure"],
                       "precision": e.metrics["precision"],
                       "rappel": e.metrics["recall"],
                       "PSNR": 10*np.log10(1/e.metrics["Square_error"])}, ignore_index=True)
        df.to_csv(r"Statistics/rec.csv")
        print(df)

        boxplot = df.boxplot(column=['precision'], by=['video', 'model'])


    def threshold_boxplot(self):
        #ranges = list(range(1, 20)) + list(range(20, 101, 5))
        ranges = list(range(5, 101, 5))
        df = pd.DataFrame(columns=["model", "video", "square_error", "fmeasure", "precision", "rappel", "PSNR", "Threshold"])
        video = "office"
        model = "recursive"
        for e in self.all_runs:
            if e.dataset["file"][9:] == video: #and e.model["model"] == model:
                for i in ranges:
                    df = df.append({"model": e.model["model"],
                       "video": e.dataset["file"][9:],
                       "square_error": e.metrics["Square_error"],
                       "fmeasure": e.metrics["fmeasure-t"+str(i)][0],
                       "precision": e.metrics["fmeasure-t"+str(i)][1],
                       "rappel": e.metrics["fmeasure-t"+str(i)][2],
                       "PSNR": 10*np.log10(1/e.metrics["Square_error"]),
                       "Threshold": i}, ignore_index=True)
        #df.to_csv(r"Statistics/rec.csv")
        print(df)

        boxplot = df.boxplot(column=['fmeasure'], by=['Threshold', 'model'])


    def variations_boxplot(self):
        ranges = range(5,201)
        videos = ["highway", "pedestrians", "PETS", "office"]
        lines = []
        for v in videos:
            lines.append(self.extract_data_from_video(video=v))

        legend = False
        for l in lines:
            if not legend:
                plt.plot(l[0], linewidth=1, color='b', label="Séquence unique")  # mean curve.
                legend = True
            else:
                plt.plot(l[0], linewidth=1, color='b')  # mean curve.
            plt.fill_between(ranges, l[1], l[2], color='b', alpha=.1)  # std curves.

        overall = self.extract_data_from_video("baseline")
        plt.plot(overall[0], linewidth=2, color='purple', label="Moyenne des séquences")  # mean curve.
        #plt.fill_between(ranges, overall[1], overall[2], color='r', alpha=.1)  # std curves.

        plt.xlabel("Nombre d'images évaluées par séquence")
        plt.ylabel("Différence de Fmeasure avec l'évaluation complète")
        plt.hlines([0.005, -0.005, 0.01, -0.01], 0, 200, colors='g', linestyles='dashed')
        plt.hlines([0], 0, 200, colors='g', linestyles='dotted')
        plt.vlines([75], -0.03, 0.03, colors='red', linestyles='solid', linewidth=2, label="Valeur sélectionnée")
        plt.ylim(-0.025,0.025)
        plt.legend()

        plt.show()

    def extract_data_from_video(self, video=""):
        ranges = range(5,201)
        data = []
        for e in self.all_runs:
            if video in e.dataset["file"]:
                res = np.asarray(e.metrics["fmeasure-nbimgs"])
                data.append(res[:, 0] - e.metrics["fmeasure"])

        data = np.asarray(data)
        smooth_path = data.mean(axis=0)
        under_line = data.min(axis=0)
        over_line = data.max(axis=0)

        print(smooth_path)

        path = np.zeros(201)
        path[5:] = smooth_path
        smooth_path = np.ma.masked_where(path == 0, path)
        return smooth_path, under_line, over_line

    def array_display(self, file=None):
        se = []
        fm = []
        for e in self.all_runs:
            if (file is None or file == e.dataset["file"]) and e.metadata["seed"] <= 100:
                se.append(e.metrics["Square_error"])
                fm.append(e.metrics["fmeasure"])
        print(min(se), max(se), np.mean(se), np.std(se), se)
        print(min(fm), max(fm), np.mean(fm), np.std(fm), fm)

        #fm = se
        mean = np.mean(fm)
        top = np.max([mean - min(fm), max(fm) - mean])
        cat = np.zeros(9)
        for i in fm:
            idx = int((i - mean + top)/(2*top)*9)
            if idx == 9: idx = 8
            cat[idx] += 1
        print(cat)


        # Distance to the mean:
        for i in range(22,21):
            res = self.select(i, fm, [], [], fm)
            print(i, "mean :", np.mean(res), "variance :", np.var(res))

    def select(self, n, array, elems, result, ref):
        if n == 0:
            t2, p2 = stats.ttest_ind(elems, ref)
            result.append(p2)
        else:
            for i in range(len(array)):
                result = self.select(n-1, array[i+1:], elems+[array[i]], result, ref)
        return result

if __name__ == '__main__':
    sr = Statistics()
    sr.open_folder(sr.folder_path)
    # sr.generate_csv()
    # sr.threshold_boxplot()
    # sr.cdnet_graph()
    # sr.surface_graph()
    sr.variations_boxplot()
    # sr.all_graph()
    # sr.array_display(file="baseline/PETS2006")
    # plt.show()

