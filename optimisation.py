import itertools
import os
import numpy as np
import multiprocessing as mp

from Code.Parameters import Parameters
from Code.execution import Execution

import pandas as pd
from hyperopt import hp, fmin, tpe, space_eval, Trials

np.set_printoptions(threshold=np.inf)

run_nb = 0

class SimulationRun:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "Optimisation")

    def create(self, args):
        global run_nb
        self.current_path = os.path.join(self.folder_path, "step"+str(run_nb))
        os.makedirs(self.current_path, exist_ok=True)

        exclusion_list = ["intermittentObjectMotion", "lowFramerate", "PTZ", "badWeather", "cameraJitter", "nightVideos", "shadow", "thermal", "dynamicBackground", "shadow", "turbulence"]

        videos_files = []
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in categories:
            if cat not in exclusion_list:
                elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
                for elem in elements:
                    videos_files.append(os.path.join(cat, elem))

        videos_files = [videos_files[2]]
        alpha_start, alpha_end, sigma_start, sigma_end = args

        for v in videos_files:
            for i in range(18, 19):
                for j in range(0, 8):
                    for k in range(20, 21):
                        exec = Execution()
                        exec.metadata = {"name": ""+v.replace("/", "_").replace("\\", "_")+str(i)+"n-"+str(k)+"p-"+str(j+1), "seed": j+1}
                        exec.dataset = {"type": "tracking", "file": v, "nb_images_evals": 75, "width": k, "height": k}
                        # exec.model = {"model": "standard", "nb_epochs": 100, "width": i, "height": i}
                        exec.model = {"model": "standard", "nb_epochs": 100, "width": i, "height": i,
                                      "alpha_start": alpha_start, "alpha_end": alpha_end,
                                      "sigma_start": sigma_start, "sigma_end": sigma_end}
                        self.all_runs.append(exec)

    def save(self):
        for e in self.all_runs:
            e.save(self.current_path)

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
        pool.starmap(Execution.full_simulation, zip(self.all_runs, itertools.repeat(self.current_path)))
        pool.close()
        pool.join()


def evaluate(args):
    global run_nb
    if run_nb > 0:
        save_optimisation()
    # pic_dim, neuron_nbr, epochs = args
    # params = Parameters({"pictures_dim": [int(pic_dim), int(pic_dim)], "neurons_nbr": (int(neuron_nbr), int(neuron_nbr)), "epochs_nbr": int(epochs)})
    alpha_start, alpha_end, sigma_start, sigma_end = args
    params = Parameters({"alpha_start": alpha_start, "alpha_end": alpha_end, "sigma_start": sigma_start, "sigma_end": sigma_end})
    print("Tested params : {}".format(params.data))

    sr = SimulationRun()
    sr.create(args)
    sr.compute(8)
    sr.all_runs = []
    sr.open_folder(sr.current_path)

    res = []
    for i in sr.all_runs:
        res.append(i.metrics["fmeasure"])
    fitness = 1 - np.mean(np.asarray(res))

    run_nb += 1

    print("Measured fitness : {}".format([fitness, 1-fitness]))
    return fitness

def save_optimisation():
    full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results[:len(tpe_trials.results)-1]] + [0],
                                 'iteration': tpe_trials.idxs_vals[0]['alpha_start'],
                                 'alpha_start': tpe_trials.idxs_vals[1]['alpha_start'],
                                 'alpha_end': tpe_trials.idxs_vals[1]['alpha_end'],
                                 'sigma_start': tpe_trials.idxs_vals[1]['sigma_start'],
                                 'sigma_end': tpe_trials.idxs_vals[1]['sigma_end']})
    full_results.to_csv(os.path.join("Statistics", "optimisation", "alpha_sigma_pedestrians.csv"))


if __name__ == '__main__':
    tpe_trials = Trials()
    # space_init = (hp.quniform('picture_dims', 5, 20, 1), hp.quniform('neurons_nbr', 8, 16, 1), hp.quniform('nb_epochs', 30, 100, 10))
    space_learning = (hp.uniform('alpha_start', 0, 1), hp.uniform('alpha_end', 0, 1), hp.uniform('sigma_start', 0, 1), hp.uniform('sigma_end', 0, 1))
    best = fmin(evaluate, space_learning, algo=tpe.suggest, trials=tpe_trials, max_evals=1000)
    save_optimisation()

    # full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
    #                              'iteration': tpe_trials.idxs_vals[0]['picture_dims'],
    #                              'picture_dims': tpe_trials.idxs_vals[1]['picture_dims'],
    #                              'neurons_nbr': tpe_trials.idxs_vals[1]['neurons_nbr'],
    #                              'nb_epochs': tpe_trials.idxs_vals[1]['nb_epochs']})
    # full_results.to_csv(os.path.join(output_path, "sizing_hyperparameters.csv"))

    # print(best)
    # print(space_eval(space_learning, best))
