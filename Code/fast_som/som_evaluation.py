import itertools

import pandas as pd
import multiprocessing as mp

from Code.Parameters import Parameters, Variable
from Code.fast_som.GreedySOM import GreedySOM
from Code.fast_som.StandardSOM import StandardSOM
from Data.Generated_Data import sierpinski_carpet, uniform


# PARAMETERS
som_algs = {"standard": StandardSOM, "greedy": GreedySOM}
data_types = {"uniform2D": uniform(500, 2), "uniform3D": uniform(500, 3)}
nb_epochs = 20


def evaluate_on_all(params):
    algo, d_type = params
    res = {"algorithm": algo, "data_type": d_type}
    res.update(evaluate_learning(som_algs[algo], data_types[d_type]))
    res.update(evaluate_reconstruction(som_algs[algo], data_types[d_type]))
    return res


def evaluate_learning(SOM, data_type):
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data_type,
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
    som = SOM(inputs)
    som.run()
    return {"MSQE_L": som.mean_square_quantization_error(), "MSDtN": som.mean_square_distance_to_neighbour()}


def evaluate_reconstruction(SOM, data_type):
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data_type,
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
    st_som = StandardSOM(inputs)
    st_som.run()

    som = SOM(inputs)
    som.neurons = st_som.neurons
    return {"MSQE_R": som.mean_square_quantization_error()}


if __name__ == '__main__':
    pool = mp.Pool(8)
    results = pool.starmap(evaluate_on_all, zip(list(itertools.product(som_algs, data_types))))
    pool.close()
    pool.join()

    full_results = pd.DataFrame(results)
    print(full_results)
