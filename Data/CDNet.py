import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

from Code.SOM import *
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
    original = data.image
    reconstructed = data.reconstruct(som.get_reconstructed_data())
    som_image = data.reconstruct(som.get_neural_list(), size=som.neurons_nbr)
    difference = ImageChops.difference(original, Image.fromarray(reconstructed))
    difference = np.asarray(difference)
    difference = np.sum(difference, axis=2)
    difference = np.divide(difference, 255*3)
    print(np.mean(np.square(difference)))
    if plot is None:
        plot = []
        plt.subplot(2, 2, 1)
        plot.append(plt.imshow(original))
        plt.subplot(2, 2, 2)
        plot.append(plt.imshow(reconstructed))
        plt.subplot(2, 2, 3)
        plot.append(plt.imshow(som_image))
        plt.subplot(2, 2, 4)
        plot.append(plt.imshow(difference))
    else:
        plot[1].set_data(reconstructed)
        plot[2].set_data(som_image)
        plot[3].set_data(difference)
    plt.pause(0.1)
    plt.draw()
plt.waitforbuttonpress()
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
