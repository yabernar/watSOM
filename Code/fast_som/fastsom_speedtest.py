import itertools
import os
import time

import pandas as pd
import numpy as np
import multiprocessing as mp

from PIL import Image

from Code.Parameters import Parameters, Variable

from Code.fast_som.FourCornersSOM import FourCornersSOM
from Code.fast_som.FourCornersSOM_Hex import FourCornersSOM_Hex
from Code.fast_som.HexSOM import HexSOM
from Code.fast_som.GreedyToroidalSOM import GreedyToroidalSOM
from Code.fast_som.SwarmSOM import SwarmSOM
from Code.fast_som.GreedySOM import GreedySOM
from Code.fast_som.StandardSOM import StandardSOM
from Data.Generated_Data import sierpinski_carpet, uniform
from Data.Generated_from_shape import GenerateFromShape
from Data.Mosaic_Image import MosaicImage

# PARAMETERS

# som_algs = {"standard": StandardSOM, "greedy": GreedySOM, "swarm": SwarmSOM, "fourCorners": FourCornersSOM, "greedyToroidal": GreedyToroidalSOM}
from Data.Pixels_from_Image import PixelsFromImage
from Data.Spoken_digits_dataset import SpokenDigitsDataset

som_algs = {"Fast-SOM": FourCornersSOM, "Fast-Hex": FourCornersSOM_Hex}
data_types = {"spokenDigits": SpokenDigitsDataset("/users/yabernar/workspace/watSOM/Data/FSDD/recordings", 1000).get_data(),
              "images": MosaicImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"), Parameters({"pictures_dim": [10, 10]})).get_data(),
              "uniform3D": uniform(1000, 3),
              "pixel_colors": PixelsFromImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"), 1000).get_data(),
              "catShape": GenerateFromShape(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/cat-silhouette.png"), 1000).get_data(),
              "uniform2D": uniform(1000, 2)}
neuron_numbers = {"12": (12, 12), "medium": (32, 32), "rectangular": (48, 24)}
n_nbr = (12, 12)
nb_epochs = 10


def regenerate_database():
    global data_types
    data_types = {"spokenDigits": SpokenDigitsDataset("/users/yabernar/workspace/watSOM/Data/FSDD/recordings", 1000).get_data(),
                  "images": MosaicImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"),
                                        Parameters({"pictures_dim": [10, 10]})).get_data(),
                  "uniform3D": uniform(1000, 3),
                  "pixel_colors": PixelsFromImage(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/Elijah.png"), 1000).get_data(),
                  "catShape": GenerateFromShape(Image.open("/users/yabernar/workspace/watSOM/Code/fast_som/cat-silhouette.png"), 1000).get_data(),
                  "uniform2D": uniform(1000, 2)}


def evaluate_on_all(params, neurons):
    d_type, algo = params
    res = {"algorithm": algo, "data_type": d_type, "neurons": neurons}
    res.update(evaluate_learning(som_algs[algo], data_types[d_type], neurons))
    # print(algo, d_type, "finished learning.")
    # res.update(evaluate_reconstruction(som_algs[algo], data_types[d_type]))
    # print(algo, d_type, "finished.")
    return res


def evaluate_learning(SOM, data_type, neurons_nbr):
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data_type,
                         "neurons_nbr": (neurons_nbr, neurons_nbr),
                         "epochs_nbr": nb_epochs})
    som = SOM(inputs)
    som.run()
    nb_comp = np.mean(som.nbr_quad_dist)
    return {"nbr_comparisons": nb_comp, "percentage_calculated": nb_comp/(neurons_nbr**2)*100}


def evaluate_reconstruction(SOM, data_type):
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data_type,
                         "neurons_nbr": n_nbr,
                         "epochs_nbr": nb_epochs})
    st_som = StandardSOM(inputs)
    st_som.run()

    som = SOM(inputs)
    som.neurons = st_som.neurons
    return {"MSQE_R": som.mean_square_quantization_error()}


if __name__ == '__main__':
    start = time.time()
    full_results = None

    for n in range(52, 101, 2):
        for i in range(1):
            pool = mp.Pool(8)
            results = pool.starmap(evaluate_on_all, zip(list(itertools.product(data_types, som_algs)), itertools.repeat(n)))
            pool.close()
            pool.join()
            if full_results is None:
                full_results = pd.DataFrame(results)
            else:
                tmp = pd.DataFrame(results)
                full_results = pd.concat([full_results, tmp], ignore_index=True)
            regenerate_database()
            # print("Test number", i+1, "finished.")
        print(n, "Neurons finished.")

    end = time.time()
    print("Executed in " + str(end - start) + " seconds.")

    print(full_results)
    full_results.to_csv(os.path.join("/users/yabernar/Documents/Quick SOM Biblio/Results/", "fast-eval2.csv"))

# 10x10 SOM, 10 epochs, 10x10 img
#        MSDtN    MSQE_L       algorithm     data_type
# 0   0.001993  0.000730        standard     uniform2D
# 1   0.001998  0.000708     fourCorners     uniform2D
# 2   0.002451  0.000680  hexFourCorners     uniform2D
# 3   0.002448  0.000683          hexSOM     uniform2D
# 4   0.001347  0.000438        standard      catShape
# 5   0.001412  0.000378     fourCorners      catShape
# 6   0.001699  0.000361  hexFourCorners      catShape
# 7   0.001630  0.000412          hexSOM      catShape
# 8   0.004316  0.003320        standard     uniform3D
# 9   0.004298  0.003323     fourCorners     uniform3D
# 10  0.005236  0.003182  hexFourCorners     uniform3D
# 11  0.004789  0.003158          hexSOM     uniform3D
# 12  0.001314  0.000182        standard  pixel_colors
# 13  0.001332  0.000210     fourCorners  pixel_colors
# 14  0.001502  0.000202  hexFourCorners  pixel_colors
# 15  0.001437  0.000183          hexSOM  pixel_colors
# 16  0.005213  0.026962        standard  spokenDigits
# 17  0.004719  0.028545     fourCorners  spokenDigits
# 18  0.005800  0.027736  hexFourCorners  spokenDigits
# 19  0.006099  0.026871          hexSOM  spokenDigits
# 20  0.001887  0.002029        standard        images
# 21  0.001863  0.002049     fourCorners        images
# 22  0.002126  0.001977  hexFourCorners        images
# 23  0.002163  0.001954          hexSOM        images

# 32x32 SOM
#        MSDtN    MSQE_L       algorithm     data_type
# 0   0.000181  0.000311        standard     uniform2D
# 1   0.000180  0.000311     fourCorners     uniform2D
# 2   0.000222  0.000300  hexFourCorners     uniform2D
# 3   0.000222  0.000302          hexSOM     uniform2D

# 4   0.000126  0.000196        standard      catShape
# 5   0.000118  0.000230     fourCorners      catShape
# 6   0.000145  0.000210  hexFourCorners      catShape
# 7   0.000151  0.000182          hexSOM      catShape

# 8   0.000372  0.002689        standard     uniform3D
# 9   0.000360  0.002748     fourCorners     uniform3D
# 10  0.000451  0.002568  hexFourCorners     uniform3D
# 11  0.000419  0.002656          hexSOM     uniform3D

# 12  0.000110  0.000125        standard  pixel_colors
# 13  0.000101  0.000131     fourCorners  pixel_colors
# 14  0.000121  0.000119  hexFourCorners  pixel_colors
# 15  0.000128  0.000115          hexSOM  pixel_colors

# 16  0.000569  0.025082        standard  spokenDigits
# 17  0.000534  0.029654     fourCorners  spokenDigits
# 18  0.000636  0.028208  hexFourCorners  spokenDigits
# 19  0.000650  0.024782          hexSOM  spokenDigits

# 20  0.000161  0.001761        standard        images
# 21  0.000160  0.001821     fourCorners        images
# 22  0.000196  0.001782  hexFourCorners        images
# 23  0.000202  0.001769          hexSOM        images
