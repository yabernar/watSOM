import os
from PIL import Image
import matplotlib.pyplot as plt

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM
from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14"

categories = sorted(os.listdir(path + "/dataset"), key=str.lower)
elements = sorted(os.listdir(path + "/dataset/" + categories[1]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[1] + "/" + elements[1] + "/input/"
temporal_ROI = (570, 2050)
images_list = []
parameters = Parameters({"pictures_dim": [5, 5]})
plot = None
img = Image.open(chosen_path + 'in{0:06d}.jpg'.format(1))
data = MosaicImage(img, parameters)

nb_epochs = 20
inputs_SOM = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data.get_data(),
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
som = SOM(inputs_SOM)
for i in range(nb_epochs):
    print('Epoch ', i)
    som.run_epoch()
    if plot is None:
        plot = plt.imshow(data.reconstruct(som.get_reconstructed_data()))
    else:
        plot.set_data(data.reconstruct(som.get_neural_list(), size=som.neurons_nbr))
    plt.pause(0.1)
    plt.draw()

# for i in range(temporal_ROI[0], temporal_ROI[1]+1):
#     print('Opening image {0:06d}'.format(i))
#     img = Image.open(chosen_path+'in{0:06d}.jpg'.format(i))
#     data = MosaicImage(img, parameters)
#     if plot is None:
#         plot = plt.imshow(img)
#     else:
#         plot.set_data(img)
#     plt.pause(0.1)
#     plt.draw()
