from Code.SOM import *


# model : class of the used model
# parameters : dictionary with the necessary parameters set
# evaluation : the function which computes the fitness
# seed : integer
def evaluate(model, parameters, evaluation):
    m = model(parameters)
    m.run()
    return evaluation(m)


if __name__ == '__main__':
    np.random.seed(0)
    nb_epochs = 20
    inputs = {"alpha": Parameter(start=0.6, end=0.05, nb_steps=nb_epochs),
              "sigma": Parameter(start=0.5, end=0.001, nb_steps=nb_epochs),
              "data": sierpinski_carpet(200),
              # "data": uniform(200, 5),  # 5 dimensional uniform data
              "neurons_nbr": (5, 5),
              "epochs_nbr": nb_epochs}
    a = evaluate(SOM, inputs, SOM.square_error)
    # a = evaluate(SOM, inputs, SOM.mean_error)
    print(a)
