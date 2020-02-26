import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

from Code.FastSOM import FastSOM
from Code.Parameters import Parameters, Variable
from Code.SOM import SOM, manhattan_distance
from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14"
path2 = "/users/yabernar/workspace/aweSOM/Data/images/tracking/ducks/"

categories = sorted(os.listdir(path + "/dataset"), key=str.lower)
elements = sorted(os.listdir(path + "/dataset/" + categories[3]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[3] + "/" + elements[2]
temporal_ROI = (400, 1700)
plot = None
bkg = Image.open(chosen_path + "/input/" + 'in{0:06d}.jpg'.format(472))
# bkg = Image.open(path + 'ducks{0:05d}.png'.format(1))

############
# LEARNING #
############
pictures_dim = [16, 16]
parameters = Parameters({"pictures_dim": pictures_dim})
data = MosaicImage(bkg, parameters)
nb_epochs = 50
data.sliding_window()
inputs_SOM = Parameters({"alpha": Variable(start=0.5, end=0.25, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.1, end=0.03, nb_steps=nb_epochs),
                         "data": data.get_data(),
                         "neurons_nbr": (20, 20),
                         "epochs_nbr": nb_epochs})
data.fixed_window()
som = FastSOM(inputs_SOM)
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

initial_map = som.get_all_winners()
plt.waitforbuttonpress()
plot = None

############
# TRACKING #
############
for i in range(temporal_ROI[0], temporal_ROI[1]):
    i *= 1
    print('Image ', i)
    current = Image.open(chosen_path + "/input/in{0:06d}.jpg".format(i))
    truth = Image.open(chosen_path + "/groundtruth/gt{0:06d}.png".format(i))
    # current = Image.open(path2 + "ducks{0:05d}.png".format(i))

    bgs_difference = ImageChops.difference(bkg, current).convert('L')

    new_data = MosaicImage(current, parameters)
    som.set_data(new_data.get_data())
    winners = som.get_all_winners()
    diff_winners = np.zeros(winners.shape)
    for j in range(len(winners)):
        diff_winners[j] = manhattan_distance(np.asarray(winners[j]), np.asarray(initial_map[j]))
    # diff_winners = np.subtract(initial_map, winners)
    # diff_winners[diff_winners == 0] = -1
    # diff_winners = np.add(diff_winners, 1)
    diff_winners = diff_winners.reshape(data.nb_pictures)
    diff_winners = np.kron(diff_winners, np.ones((pictures_dim[0], pictures_dim[1])))
    diff_winners *= 30
    diff_winners = Image.fromarray(diff_winners).convert('L')

    reconstructed = data.reconstruct(som.get_reconstructed_data(winners))
    som_difference = ImageChops.difference(Image.fromarray(reconstructed), current).convert('L')
    som_difference = ImageChops.multiply(som_difference, diff_winners)

    som_difference.save("/users/yabernar/workspace/watSOM/Results/DNF/saliency"+str(i)+".png")
#    diff_winners.save("/users/yabernar/workspace/watSOM/Results/diff_winners"+str(i)+".png")

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
        plot[2].set_data(diff_winners)
        plot[3].set_data(som_difference)
#        plot[3].set_data(truth)
    plt.pause(0.02)
    plt.draw()
plt.waitforbuttonpress()