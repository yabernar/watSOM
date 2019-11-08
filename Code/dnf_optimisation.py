import itertools
import os
import numpy as np
import multiprocessing as mp

import pandas as pd
from PIL import Image
from hyperopt import hp, fmin, tpe, space_eval, Trials

from Code.Parameters import Parameters
from Code.cdnet_evaluation import Comparator
from Code.dnf_generator import DNFGenerator
from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

input_path = "/users/yabernar/GrosDisque/CDNET14/saliency_as/supplements/"
output_path = "/users/yabernar/GrosDisque/CDNET14/optimisation"
Processing = DNFGenerator


def evaluate(args):
    # pic_dim, neuron_nbr, epochs = args
    # params = Parameters({"pictures_dim": [int(pic_dim), int(pic_dim)], "neurons_nbr": (int(neuron_nbr), int(neuron_nbr)), "epochs_nbr": int(epochs)})
    alpha_start, alpha_end, sigma_start, sigma_end = args
    params = Parameters({"alpha_start": alpha_start, "alpha_end": alpha_end, "sigma_start": sigma_start, "sigma_end": sigma_end})
    print("Tested params : {}".format(params.data))

    all_videos = []
    categories_list = [1, 3]
    categories = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))], key=str.lower)
    for cat in list(categories[i] for i in categories_list):
        elements = sorted([d for d in os.listdir(os.path.join(input_path, cat)) if os.path.isdir(os.path.join(input_path, cat, d))], key=str.lower)
        for elem in elements:
            all_videos.append(os.path.join(cat, elem))

    pool = mp.Pool(8)
    results = pool.starmap(process_video, zip(all_videos, itertools.repeat(params)))
    print(results)
    pool.close()
    pool.join()

    fitness = np.mean(np.asarray(results))

    print("Measured fitness : {}\n".format(fitness))
    return fitness


def process_video(video, params):
    os.makedirs(os.path.join(output_path, 'results', video), exist_ok=True)
    current_path = os.path.join(input_path, video, "saliency")
    pr = Processing(current_path, os.path.join(output_path, 'results', video), os.path.join(output_path, 'supplements', video), parameters=params)
    pr.optimize()


if __name__ == '__main__':
    tpe_trials = Trials()
    # space_init = (hp.quniform('picture_dims', 5, 20, 1), hp.quniform('neurons_nbr', 8, 16, 1), hp.quniform('nb_epochs', 30, 100, 10))
    space_learning = (hp.uniform('alpha_start', 0, 1), hp.uniform('alpha_end', 0, 1), hp.uniform('sigma_start', 0, 1), hp.uniform('sigma_end', 0, 1))
    best = fmin(evaluate, space_learning, algo=tpe.suggest, trials=tpe_trials, max_evals=120)

    full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                'iteration': tpe_trials.idxs_vals[0]['alpha_start'],
                                'alpha_start': tpe_trials.idxs_vals[1]['alpha_start'],
                                'alpha_end': tpe_trials.idxs_vals[1]['alpha_end'],
                                'sigma_start': tpe_trials.idxs_vals[1]['sigma_start'],
                                'sigma_end': tpe_trials.idxs_vals[1]['sigma_end']})
    full_results.to_csv(os.path.join(output_path, "basic_alpha_sigma.csv"))

    # full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
    #                              'iteration': tpe_trials.idxs_vals[0]['picture_dims'],
    #                              'picture_dims': tpe_trials.idxs_vals[1]['picture_dims'],
    #                              'neurons_nbr': tpe_trials.idxs_vals[1]['neurons_nbr'],
    #                              'nb_epochs': tpe_trials.idxs_vals[1]['nb_epochs']})
    # full_results.to_csv(os.path.join(output_path, "sizing_hyperparameters.csv"))

    # print(best)
    # print(space_eval(space_learning, best))
