import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM
from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14"
path2 = "/users/yabernar/workspace/aweSOM/Data/images/tracking/dogs/"

categories = sorted(os.listdir(path + "/dataset"), key=str.lower)
elements = sorted(os.listdir(path + "/dataset/" + categories[1]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[1] + "/" + elements[1]
temporal_ROI = (1, 3200)
plot = None
bkg = Image.open(chosen_path + "/input/" + 'in{0:06d}.jpg'.format(1))
# bkg = Image.open(path2 + 'dogs{0:05d}.png'.format(1))

############
# LEARNING #
############
pictures_dim = [10, 10]
parameters = Parameters({"pictures_dim": pictures_dim})
data = MosaicImage(bkg, parameters)
nb_epochs = 5
inputs_SOM = Parameters({"alpha": Variable(start=0.5, end=0.25, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.1, end=0.001, nb_steps=nb_epochs),
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
    difference = ImageChops.difference(original, Image.fromarray(reconstructed)).convert('L')
    difference = np.asarray(difference)
    # difference = np.sum(difference, axis=2)
    # difference = np.divide(difference, 255*3)
    difference = np.divide(difference, 255)
    print(np.mean(np.square(difference)))
    if plot is None:
        plot = []
        plt.subplot(2, 2, 1)
        plot.append(plt.imshow(original))
        plt.subplot(2, 2, 2)
        plot.append(plt.imshow(reconstructed, cmap='gray'))
        plt.subplot(2, 2, 3)
        plot.append(plt.imshow(som_image, cmap='gray'))
        plt.subplot(2, 2, 4)
        plot.append(plt.imshow(difference, cmap='gray'))
    else:
        plot[1].set_data(reconstructed)
        plot[2].set_data(som_image)
        plot[3].set_data(difference)
    plt.pause(0.02)
    plt.draw()
plt.waitforbuttonpress()
plot = None

############
# TRACKING #
############
for i in range(temporal_ROI[0], temporal_ROI[1]):
    i *= 10
    print('Image ', i)
    current = Image.open(chosen_path + "/input/in{0:06d}.jpg".format(i))
    truth = Image.open(chosen_path + "/groundtruth/gt{0:06d}.png".format(i))
    # current = Image.open(path2 + "dogs{0:05d}.png".format(i))

    bgs_difference = ImageChops.difference(bkg, current).convert('L')

    new_data = MosaicImage(current, parameters)
    som.set_data(new_data.get_data())
    reconstructed = data.reconstruct(som.get_reconstructed_data())
    som_difference = ImageChops.difference(current, Image.fromarray(reconstructed)).convert('L')

    if plot is None:
        plot = []
        plt.subplot(2, 2, 1)
        plot.append(plt.imshow(current))
        plt.subplot(2, 2, 2)
        plot.append(plt.imshow(reconstructed))
        plt.subplot(2, 2, 3)
        plot.append(plt.imshow(bgs_difference, cmap='gray'))
        plt.subplot(2, 2, 4)
        plot.append(plt.imshow(truth))
    else:
        plot[0].set_data(current)
        plot[1].set_data(reconstructed)
        plot[2].set_data(bgs_difference)
        plot[3].set_data(som_difference)
#        plot[3].set_data(truth)
    plt.pause(0.02)
    plt.draw()
plt.waitforbuttonpress()