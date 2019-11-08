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
    tau_dt, h, gi, Ap, Sp = args
    params = Parameters({"tau_dt": tau_dt, "h": h, "gi": gi, "excitation_amplitude": Ap, "excitation_sigma": Sp})
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
    current_path = os.path.join(input_path, video, "supplements", "saliency")
    pr = Processing(current_path, os.path.join(output_path, 'results', video), os.path.join(output_path, 'supplements', video), parameters=params)
    pr.optimize()


if __name__ == '__main__':
    tpe_trials = Trials()

    space_dnf = (hp.uniform('tau_dt', 0.0001, 1), hp.uniform('h', -1, 0), hp.uniform('gi', 0, 10), hp.uniform('excitation_amplitude', 0.0001, 5), hp.uniform('excitation_sigma', 0.0001, 1))
    best = fmin(evaluate, space_dnf, algo=tpe.suggest, trials=tpe_trials, max_evals=120)

    full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                'iteration': tpe_trials.idxs_vals[0]['tau_dt'],
                                'tau_dt': tpe_trials.idxs_vals[1]['tau_dt'],
                                'h': tpe_trials.idxs_vals[1]['h'],
                                'gi': tpe_trials.idxs_vals[1]['gi'],
                                'excitation_amplitude': tpe_trials.idxs_vals[1]['excitation_amplitude'],
                                'excitation_sigma': tpe_trials.idxs_vals[1]['excitation_sigma']})
    full_results.to_csv(os.path.join(output_path, "dnf_cdnet.csv"))

    print(best)
    print(space_eval(space_dnf, best))
