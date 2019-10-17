import itertools
import os
import numpy as np
import multiprocessing as mp
from PIL import Image
from hyperopt import hp, fmin, tpe, space_eval, Trials

from Code.cdnet_evaluation import Comparator
from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
output_path = "/users/yabernar/GrosDisque/CDNET14/optimisation"
Processing = SaliencyGenerator


def evaluate(args):
    threshold = args
    print("Tested threshold : {}".format(threshold))
    all_videos = []
    categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
    for cat in categories:
        elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
        for elem in elements:
            all_videos.append(os.path.join(cat, elem))

    pool = mp.Pool(8)
    pool.starmap(process_video, zip(all_videos, itertools.repeat(threshold)))
    pool.close()
    pool.join()

    cmp = Comparator()
    fitness = cmp.evaluate_all(cdnet_path, output_path)
    print("Measured fitness : {}\n".format(fitness))
    return 1 - fitness


def process_video(video, threshold):
    os.makedirs(os.path.join(output_path, 'results', video), exist_ok=True)
    current_path = os.path.join(cdnet_path, video)
    roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
    temporal_roi = (int(roi_file[0]), int(roi_file[1]))
    mask_roi = Image.open(os.path.join(current_path, "ROI.png"))
    pr = Processing(os.path.join(current_path, "input"), os.path.join(output_path, 'results', video), os.path.join(output_path, 'supplements', video),
               temporal_roi, mask_roi, threshold=threshold)
    pr.optimize()


if __name__ == '__main__':
    tpe_trials = Trials()
    space = hp.quniform('threshold', 0, 35, 1)
    best = fmin(evaluate, space, algo=tpe.suggest, trials=tpe_trials, max_evals=50)

    print(best)
    print(space_eval(space, best))
