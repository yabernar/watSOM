import itertools

import pandas as pd
import multiprocessing as mp

from PIL import Image

from Code.Parameters import Parameters, Variable
from Code.fast_som.FourCornersSOM import FourCornersSOM
from Code.fast_som.GreedyToroidalSOM import GreedyToroidalSOM
from Code.fast_som.SwarmSOM import SwarmSOM
from Code.fast_som.GreedySOM import GreedySOM
from Code.fast_som.StandardSOM import StandardSOM
from Data.Generated_Data import sierpinski_carpet, uniform
from Data.Mosaic_Image import MosaicImage

# PARAMETERS

# som_algs = {"standard": StandardSOM, "greedy": GreedySOM, "swarm": SwarmSOM, "fourCorners": FourCornersSOM, "greedyToroidal": GreedyToroidalSOM}
som_algs = {"standard": StandardSOM, "fourCorners": FourCornersSOM}
data_types = {"uniform2D": uniform(500, 2), "uniform3D": uniform(500, 3),
              "images": MosaicImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"), Parameters({"pictures_dim": [10, 10]})).get_data()}
neuron_number = {"small": (10, 10), "medium": (32, 32), "rectangular": (64, 16)}
nb_epochs = 20


def evaluate_on_all(params):
    d_type, algo = params
    res = {"algorithm": algo, "data_type": d_type}
    res.update(evaluate_learning(som_algs[algo], data_types[d_type]))
    print(algo, d_type, "finished learning.")
    res.update(evaluate_reconstruction(som_algs[algo], data_types[d_type]))
    print(algo, d_type, "finished.")
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
    results = pool.starmap(evaluate_on_all, zip(list(itertools.product(data_types, som_algs))))
    pool.close()
    pool.join()

    full_results = pd.DataFrame(results)
    print(full_results)

## 10x10 SOM
#       MSDtN    MSQE_L    MSQE_R       algorithm  data_type
# 0  0.002236  0.000520  0.000499        standard  uniform2D
# 1  0.002236  0.000520  0.000499     fourCorners  uniform2D
# 2  0.006930  0.001618  0.000527  greedyToroidal  uniform2D
# 3  0.005106  0.002665  0.002728        standard  uniform3D
# 4  0.005042  0.003269  0.003149     fourCorners  uniform3D
# 5  0.011549  0.015693  0.020495  greedyToroidal  uniform3D
# 6  0.002493  0.000827  0.000816        standard     images
# 7  0.004002  0.001224  0.001032     fourCorners     images
# 8  0.005424  0.001733  0.002265  greedyToroidal     images

## 24x24 SOM
#       MSDtN    MSQE_L    MSQE_R       algorithm  data_type
# 0  0.000353  0.000216  0.000221        standard  uniform2D
# 1  0.000353  0.000216  0.000221     fourCorners  uniform2D
# 2  0.001189  0.000683  0.000228  greedyToroidal  uniform2D
# 3  0.000847  0.001837  0.001785        standard  uniform3D
# 4  0.000910  0.006476  0.002788     fourCorners  uniform3D
# 5  0.002362  0.018406  0.018079  greedyToroidal  uniform3D
# 6  0.000446  0.000588  0.000583        standard     images
# 7  0.000694  0.001655  0.001098     fourCorners     images
# 8  0.001852  0.002310  0.002205  greedyToroidal     images

# 10x10 SOM and IMG
#       MSDtN    MSQE_L    MSQE_R    algorithm  data_type
# 0  0.002278  0.000507  0.000519     standard  uniform2D
# 1  0.002269  0.000516  0.000525  fourCorners  uniform2D
# 2  0.005134  0.002480  0.002579     standard  uniform3D
# 3  0.004739  0.002580  0.002485  fourCorners  uniform3D
# 4  0.002307  0.001851  0.001844     standard     images
# 5  0.002308  0.001824  0.001867  fourCorners     images

# 24*24 SOM and 4x4 IMG
#       MSDtN    MSQE_L    MSQE_R    algorithm  data_type
# 0  0.000359  0.000214  0.000221     standard  uniform2D
# 1  0.000360  0.000214  0.000217  fourCorners  uniform2D
# 2  0.000828  0.001780  0.001736     standard  uniform3D
# 3  0.000833  0.001885  0.002127  fourCorners  uniform3D
# 4  0.000451  0.000584  0.000587     standard     images
# 5  0.000567  0.001100  0.000729  fourCorners     images
