import itertools
import time

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
    inputs = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": sierpinski_carpet(200),
                         # "data": uniform(200, 5),  # 5 dimensional uniform data
                         "neurons_nbr": (5, 5),
                         "epochs_nbr": nb_epochs})
    # a = evaluate(SOM, inputs, SOM.square_error)
    # a = evaluate(SOM, inputs, SOM.mean_error)

    # Multiprocessor version:
    start = time.time()
    nb_runs = 4
    pool = mp.Pool(nb_runs)
    a = pool.starmap(evaluate, zip(itertools.repeat(SOM, nb_runs), itertools.repeat(inputs, nb_runs), itertools.repeat(SOM.square_error, nb_runs)))
    pool.close()
    pool.join()
    end = time.time()
    print(a)
    print("Executed in "+str(end-start)+" seconds.")




