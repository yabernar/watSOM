import itertools
import os
import numpy as np
import multiprocessing as mp

import pandas as pd
from PIL import Image
from hyperopt import hp, fmin, tpe, space_eval, Trials

from Code.Parameters import Parameters
from Code.cdnet_evaluation import Comparator
from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
output_path = "/users/yabernar/GrosDisque/CDNET14/optimisation"
Processing = SaliencyGenerator


def evaluate(args):
    pic_dim, neuron_nbr, epochs = args
    params = Parameters({"pictures_dim": [int(pic_dim), int(pic_dim)], "neurons_nbr": (int(neuron_nbr), int(neuron_nbr)), "epochs_nbr": int(epochs)})
    # alpha_start, alpha_end, sigma_start, sigma_end = args
    # params = Parameters({"alpha_start": alpha_start, "alpha_end": alpha_end, "sigma_start": sigma_start, "sigma_end": sigma_end})
    print("Tested params : {}".format(params.data))

    all_videos = []
    categories_list = [1, 3]
    categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
    for cat in list(categories[i] for i in categories_list):
        elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
        for elem in elements:
            all_videos.append(os.path.join(cat, elem))

    pool = mp.Pool(8)
    pool.starmap(process_video, zip(all_videos, itertools.repeat(params)))
    pool.close()
    pool.join()

    cmp = Comparator()
    fitness = cmp.evaluate_all(cdnet_path, output_path, categories_list)
    print("Measured fitness : {}\n".format(fitness))
    return 1 - fitness


def process_video(video, params):
    os.makedirs(os.path.join(output_path, 'results', video), exist_ok=True)
    current_path = os.path.join(cdnet_path, video)
    roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
    temporal_roi = (int(roi_file[0]), int(roi_file[1]))
    mask_roi = Image.open(os.path.join(current_path, "ROI.png"))
    pr = Processing(os.path.join(current_path, "input"), os.path.join(output_path, 'results', video), os.path.join(output_path, 'supplements', video),
                    temporal_roi, mask_roi, parameters=params)
    pr.optimize()


if __name__ == '__main__':
    tpe_trials = Trials()
    space_init = (hp.quniform('picture_dims', 5, 20, 1), hp.quniform('neurons_nbr', 8, 16, 1), hp.quniform('nb_epochs', 30, 100, 10))
    # space_learning = (hp.uniform('alpha_start', 0, 1), hp.uniform('alpha_end', 0, 1), hp.uniform('sigma_start', 0, 1), hp.uniform('sigma_end', 0, 1))
    best = fmin(evaluate, space_init, algo=tpe.suggest, trials=tpe_trials, max_evals=250)

    # full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
    #                             'iteration': tpe_trials.idxs_vals[0]['alpha_start'],
    #                             'alpha_start': tpe_trials.idxs_vals[1]['alpha_start'],
    #                             'alpha_end': tpe_trials.idxs_vals[1]['alpha_end'],
    #                             'sigma_start': tpe_trials.idxs_vals[1]['sigma_start'],
    #                             'sigma_end': tpe_trials.idxs_vals[1]['sigma_end']})
    # full_results.to_csv(os.path.join(output_path, "alpha_sigma.csv"))

    full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                 'iteration': tpe_trials.idxs_vals[0]['picture_dims'],
                                 'picture_dims': tpe_trials.idxs_vals[1]['picture_dims'],
                                 'neurons_nbr': tpe_trials.idxs_vals[1]['neurons_nbr'],
                                 'nb_epochs': tpe_trials.idxs_vals[1]['nb_epochs']})
    full_results.to_csv(os.path.join(output_path, "sizing_hyperparameters.csv"))

    # print(best)
    # print(space_eval(space_learning, best))
