import itertools
import time

from Code.DNF import *
from Code.SOM import *
import multiprocessing as mp


# model : class of the used model
# parameters : dictionary with the necessary parameters set
# evaluation : the function which computes the fitness
# seed : integer
def evaluate(model, parameters, evaluation):
    np.random.seed()
    m = model(parameters)
    m.run()
    return evaluation(m)


if __name__ == '__main__':
    nb_epochs = 20
    inputs_SOM = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                             "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                             "data": sierpinski_carpet(200),
                             # "data": uniform(200, 5),  # 5 dimensional uniform data
                             "neurons_nbr": (5, 5),
                             "epochs_nbr": nb_epochs})
    inputs_DNF = {"tau_dt": 0.92,   # entre 0 et 1
                  "h": -0.7,        # entre -1 et 0
                  "Ap": 4.0,        # entre 1 et 10
                  "Sp": 0.01,       # entre 0 et 1
                  "Am": 2.0,        # entre 1 et 10
                  "Sm": 0.033,      # entre 0 et 1
                  "gi": 50}        # entre 0 et 500

    start = time.time()
    # a = evaluate(SOM, inputs, SOM.square_error)
    # a = evaluate(SOM, inputs, SOM.mean_error)
    # a = evaluate(DNF, inputs_DNF, DNF.mean_squared_error)

    # Multiprocessor version:
    nb_runs = 4
    pool = mp.Pool(nb_runs)
    a = pool.starmap(evaluate, zip(itertools.repeat(DNF, nb_runs), itertools.repeat(inputs_DNF, nb_runs), itertools.repeat(DNF.mean_squared_error, nb_runs)))
    # a = pool.starmap(evaluate, zip(itertools.repeat(SOM, nb_runs), itertools.repeat(inputs_SOM, nb_runs), itertools.repeat(SOM.square_error, nb_runs)))
    pool.close()
    pool.join()

    end = time.time()
    print(a)
    print("Executed in " + str(end - start) + " seconds.")
