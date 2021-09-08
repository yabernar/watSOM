import itertools
import multiprocessing as mp
import numpy as np
import os

from itertools import chain

from scipy.stats import stats

from Code.execution import Execution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import pandas as pd

font = {'size'   : 24}
#font = {'size'   : 14}
#font = {'size'   : 12}

matplotlib.rc('font', **font)

class Statistics:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "EpochsNbr2")
        self.results_path = os.path.join("Statistics", "sizing_psnr")

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.light_open(full_path)
                self.all_runs.append(exec)

    def get_cdnet_videos_list(self):
        videos_files = []
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))],
                            key=str.lower)
        for cat in categories:
            elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if
                               os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
            for elem in elements:
                videos_files.append(os.path.join(cat, elem))
        return videos_files

    def cdnet_surface(self):
        os.makedirs(self.results_path, exist_ok=True)

        videos_files = self.get_cdnet_videos_list()
        for file in videos_files:
            fig = self.surface_graph(file)
            plt.savefig(os.path.join(self.results_path, "videos", file.replace("/", "_").replace("\\", "_") + ".png"))
            plt.close(fig)

        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)

        for cat in categories:
            fig = self.surface_graph(cat)
            plt.savefig(os.path.join(self.results_path, "categories", cat + ".png"))
            plt.close(fig)

        fig = self.cdnet_global_surface_graph()
        plt.savefig(os.path.join(self.results_path, "global.png"))
        plt.close(fig)

    def surface_graph(self, file=""):
        nbr_samples = 0
        surface = np.zeros((8, 7))
        for e in self.all_runs:
            if file in e.dataset["file"]:
                if e.model["width"] == 4 and e.dataset["width"] == 5:
                    nbr_samples += 1        # Couting the number of samples
                #surface[(e.model["width"]-4)//3, (e.dataset["width"]-5)//5] += e.metrics["fmeasure_threshold"][6][0]  # Seuil de 7 (commence à 1)
                surface[(e.model["width"] - 4) // 3, (e.dataset["width"] - 5) // 5] += 10*np.log10(1/e.metrics["Square_error"])

        surface /= nbr_samples

        X, Y = np.meshgrid(range(5, 36, 5), range(4, 26, 3))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim(15, 50) # 0.95 * zmin, 1.05 * zmax)
        ax.set_xlabel("Taille imagettes")
        ax.set_ylabel("Taille carte")
        ax.set_xticks(list(range(5, 36, 5)))
        ax.set_yticks(list(range(4, 26, 3)))
        ax.locator_params(axis="z", nbins=6)
        ax.invert_xaxis()
        ax.plot_surface(X, Y, surface, cmap=cm.plasma)
        return fig

    def cdnet_global_surface_graph(self):
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        global_surface = np.zeros((8,7))
        for cat in categories:
            nbr_samples = 0
            surface = np.zeros((8, 7))
            for e in self.all_runs:
                if cat in e.dataset["file"]:
                    if e.model["width"] == 4 and e.dataset["width"] == 5:
                        nbr_samples += 1        # Couting the number of samples
                    #surface[(e.model["width"]-4)//3, (e.dataset["width"]-5)//5] += e.metrics["fmeasure_threshold"][6][0]
                    surface[(e.model["width"] - 4) // 3, (e.dataset["width"] - 5) // 5] += 10*np.log10(1/e.metrics["Square_error"])
            surface /= nbr_samples
            global_surface += surface

        global_surface /= len(categories)
        print(categories)
        print(len(categories), np.argmax(global_surface), np.max(global_surface))
        X, Y = np.meshgrid(range(5, 36, 5), range(4, 26, 3))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim(15, 45) # 0.95 * zmin, 1.05 * zmax)
        ax.set_xlabel("Taille imagettes")
        ax.set_ylabel("Taille carte")
        ax.set_xticks(list(range(5, 36, 5)))
        ax.set_yticks(list(range(4, 26, 3)))
        ax.locator_params(axis="z", nbins=6)
        ax.invert_xaxis()
        ax.plot_surface(X, Y, global_surface, cmap=cm.plasma)
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

    def nb_images_per_sequence_plot(self):
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
        plt.ylabel("Différence de F-mesure avec l'évaluation complète")
        plt.hlines([0.005, -0.005, 0.01, -0.01], 0, 200, colors='g', linestyles='dashed')
        plt.hlines([0], 0, 200, colors='g', linestyles='dotted')
        plt.vlines([105], -0.03, 0.03, colors='red', linestyles='solid', linewidth=2, label="Valeur sélectionnée")
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

    def nb_epochs_plot(self):
        ranges = chain(range(5, 201, 5), range(7, 201, 5))
        x = sorted(list(ranges))
        videos = ["highway", "pedestrians", "PETS", "office"]
        lines = []
        for v in videos:
            lines.append(self.extract_data_from_video_epoch(video=v))

        legend = False
        for l in lines:
            if not legend:
                plt.plot(x, l[0], linewidth=1, color='b', label="Séquence unique")  # mean curve.
                legend = True
            else:
                plt.plot(x, l[0], linewidth=1, color='b')  # mean curve.
            plt.fill_between(x, l[1], l[2], color='b', alpha=.1)  # std curves.

        overall = self.extract_data_from_video_epoch("baseline")
        plt.plot(x, overall[0], linewidth=2, color='purple', label="Moyenne des séquences")  # mean curve.

        cleaned = self.extract_data_from_video_epoch("baseline", "PETS")
        plt.plot(x, cleaned[0], linewidth=2, color='green', label="Moyenne des séquences croissantes")  # mean curve.
        plt.hlines([max(cleaned[0]), min(cleaned[0][40:])], 0, 200, colors='g', linestyles='dotted')
        print([max(cleaned[0]), min(cleaned[0][40:])])

        print(max(cleaned[0]))


        plt.xlabel("Époques d'apprentissage")
        plt.ylabel("F-mesure")
        #plt.hlines([0.005, -0.005, 0.01, -0.01], 0, 200, colors='g', linestyles='dashed')
        #plt.hlines([0], 0, 200, colors='g', linestyles='dotted')
        #plt.vlines([105], -0.03, 0.03, colors='red', linestyles='solid', linewidth=2, label="Valeur sélectionnée")
        #plt.ylim(-0.025,0.025)
        plt.legend()

        plt.show()

    def extract_data_from_video_epoch(self, video="", exclusion="None"):
        ranges = chain(range(5, 201, 5), range(7,201,5))
        data = []
        for i in ranges:
            data.append([])
        for e in self.all_runs:
            if video in e.dataset["file"] and exclusion not in e.dataset["file"]:
                if e.model["nb_epochs"] % 5 == 0:
                    data[(e.model["nb_epochs"]//5-1)*2].append(e.metrics["fmeasure"])
                else:
                    data[((e.model["nb_epochs"]-2)//5-1)*2+1].append(e.metrics["fmeasure"])

        data = np.asarray(data)
        smooth_path = data.mean(axis=1)
        under_line = data.min(axis=1)
        over_line = data.max(axis=1)

        #print(smooth_path)

        #path = np.zeros(201)
        #path = smooth_path
        #smooth_path = np.ma.masked_where(path == 0, path)
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

    def global_threshold(self):
        ranges = chain(range(1, 20), range(20, 51, 5))
        x = sorted(list(ranges))

        overall = self.extract_data_from_video_threshold("")
        plt.plot(x, overall[:,0], linewidth=2, color='purple', label="F-mesure")  # mean curve.
        plt.plot(x, overall[:,1], linewidth=2, color='blue', label="Précision")  # mean curve.
        plt.plot(x, overall[:,2], linewidth=2, color='red', label="Rappel")  # mean curve.

        print(np.argmax(overall[:,0]), np.max(overall[:,0]))

        plt.xlabel("Seuil")
        plt.ylabel("")
        plt.legend()
        plt.show()

    def extract_data_from_video_threshold(self, video="", exclusion="None"):
        data = {}
        for e in self.all_runs:
            if video in e.metadata["name"] and exclusion not in e.dataset["file"]:
                cat = e.dataset["file"].split("/")[0]
                if cat not in data:
                    data[cat] = []
                data[cat].append(np.asarray(e.metrics["fmeasure_threshold"])[:26])
                #data.append(np.asarray(e.metrics["fmeasure_threshold"])[:26])
        res = []
        for v in data.items():
            res.append(np.asarray(v[1]).mean(axis=0))
        return np.asarray(res).mean(axis=0)

    def cdnet_threshold(self):
        neurones = ["4x4", "7x7", "10x10", "13x13", "16x16", "19x19", "22x22", "25x25"]
        pixels = ["5x5 px", "10x10 px", "15x15 px", "20x20 px", "25x25 px", "30x30 px", "35x35 px"]

        data = []
        for i in range(4, 26, 3):
            line = []
            for j in range(5, 36, 5):
                res = self.extract_data_from_video_threshold(str(i)+"n-"+str(j)+"p")
                line.append(np.argmax(res[:,0])+1)
                #line.append(np.around(np.max(res[:,0])*100, 2))
            data.append(np.asarray(line))
        data = np.asarray(data)

        fig, ax = plt.subplots()
        im = ax.imshow(data)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(pixels)))
        ax.set_yticks(np.arange(len(neurones)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(pixels)
        ax.set_yticklabels(neurones)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(neurones)):
            for j in range(len(pixels)):
                text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

        ax.set_title("Seuil optimal en fonction de la\n taille des imagettes et de la SOM")
        #ax.set_title("F-mesure avec seuil optimal en fonction\nde la taille des imagettes et de la SOM")
        fig.tight_layout()
        plt.show()

    def cdnet_stats(self, pixel_size, neuron_size, threshold):
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        global_all = []
        max_all = []
        for cat in categories:
            nbr_samples = 0
            global_cat = []
            max_cat = []
            elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
            for elem in elements:
                global_video = []
                max = 0
                for e in self.all_runs:
                    if elem in e.dataset["file"]:
                        if e.model["width"] == neuron_size and e.dataset["width"] == pixel_size:
                            global_video = e.metrics["fmeasure_threshold"][threshold-1]
                            nbr_samples += 1
                        values = np.asarray(e.metrics["fmeasure_threshold"])[:,0]
                        current_max = np.argmax(values)
                        if values[current_max] > max:
                            max = values[current_max]
                global_video = np.asarray(global_video) * 100
                max *= 100
                print(elem,"&", str(np.around(global_video[1], 1))+"\% &", str(np.around(global_video[2], 1))+"\% &", str(np.around(global_video[0], 1))+"\% &", str(np.around(max, 1))+"\% & &\\\\")
                global_cat.append(global_video)
                max_cat.append(max)
            global_cat = np.asarray(global_cat).mean(axis=0)
            max_cat = np.asarray(max_cat).mean(axis=0)
            print(cat, "&", str(np.around(global_cat[1], 1)) + "\% &", str(np.around(global_cat[2], 1)) + "\% &", str(np.around(global_cat[0], 1)) + "\% &", str(np.around(max_cat, 1)) + "\% & &\\\\")
            global_all.append(global_cat)
            max_all.append(max_cat)
        global_all = np.asarray(global_all).mean(axis=0)
        max_all = np.asarray(max_all).mean(axis=0)
        print("all &", str(np.around(global_all[1], 1)) + "\% &", str(np.around(global_all[2], 1)) + "\% &", str(np.around(global_all[0], 1)) + "\% &", str(np.around(max_all, 1)) + "\% & &\\\\")

    def randomness_stats(self):
        labels = ['', '', '', '', '', '', '', '', '']
        Fmeasure = [2,2,9,21,30,22,12,1,1]
        MSQE = [0,4,14,23,23,16,14,5,1]

        x = np.arange(len(labels))  # the label locations
        width = 0.42  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, Fmeasure, width, label='F-mesure')
        rects2 = ax.bar(x + width / 2, MSQE, width, label='MSQE')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Nombre d\'exécutions')
        ax.set_title('Variation aléatoire des SOM')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 34)

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()

if __name__ == '__main__':
    sr = Statistics()
    sr.open_folder(sr.folder_path)
    # sr.generate_csv()
    #sr.global_threshold()
    #sr.cdnet_threshold()
    #sr.cdnet_surface()
    # sr.surface_graph()
    #sr.variations_boxplot()
    #sr.nb_images_per_sequence_plot()
    sr.nb_epochs_plot()
    # sr.all_graph()
    # sr.array_display(file="baseline/PETS2006")
    # plt.show()
    #sr.cdnet_stats(20, 25, 7)
    #sr.randomness_stats()