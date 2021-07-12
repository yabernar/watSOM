import os
from PIL import Image, ImageChops, ImageOps
import matplotlib.pyplot as plt
import numpy as np

from Code.DNF_original import DNF
from Code.Parameters import Parameters, Variable
from Code.SOM import SOM, manhattan_distance
from Data.Mosaic_Image import MosaicImage

np.set_printoptions(threshold=np.inf)

video = "dogs"

path = "/users/yabernar/GrosDisque/CDNET14"
path2 = "/users/yabernar/workspace/aweSOM/Data/images/tracking/"+video+"/"
path3 = "/users/yabernar/Documents/Presentation resources/Model/base image/"

categories = sorted(os.listdir(path + "/dataset"), key=str.lower)
elements = sorted(os.listdir(path + "/dataset/" + categories[11]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[11] + "/" + elements[0]
temporal_ROI = (1, 200)
plot = None
bkg = Image.open(chosen_path + "/input/" + 'bkg.jpg')
# bkg = Image.open(path2 + video + '{0:05d}.png'.format(1))
# bkg = Image.open(path3 + "example_base.png")

############
# LEARNING #
############
pictures_dim = [10, 10]
parameters = Parameters({"pictures_dim": pictures_dim})
data = MosaicImage(bkg, parameters)
nb_epochs = 50
inputs_SOM = Parameters({"alpha": Variable(start=0.5, end=0.25, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.1, end=0.03, nb_steps=nb_epochs),
                         "data": data.get_data(),
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
som = SOM(inputs_SOM)
for i in range(nb_epochs):
    print('Epoch ', i)
    som.run_epoch()
    original = data.image
    reconstructed = Image.fromarray(data.reconstruct(som.get_reconstructed_data()))
    som_image = data.reconstruct(som.get_neural_list(), size=som.neurons_nbr)
    difference_image = ImageChops.difference(original, reconstructed).convert('L')
    difference = np.asarray(difference_image)
    # difference = np.sum(difference, axis=2)
    # difference = np.divide(difference, 255*3)
    difference = np.divide(difference, 255)
    print(np.mean(np.square(difference)))
    som_image = Image.fromarray(som_image)

    # reconstructed.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/training/reconstructed"+"{0:02d}.png".format(i))
    # som_image.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/training/som_image{0:02d}.png".format(i))
    # difference_image.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/training/difference{0:02d}.png".format(i))

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
plot = None
plt.waitforbuttonpress()
dnf_size = (data.nb_pictures[0]*data.pictures_dim[0], data.nb_pictures[1]*data.pictures_dim[0])
# dnf = DNF(dnf_size[0], dnf_size[1])

############
# TRACKING #
############
for i in range(temporal_ROI[0], temporal_ROI[1]):
    i *= 1
    print('Image ', i)
    current = Image.open(chosen_path + "/input/in{0:06d}.jpg".format(i))
    # truth = Image.open(chosen_path + "/groundtruth/gt{0:06d}.png".format(i))
    # current = Image.open(path2 + video + "{0:05d}.png".format(i))
    # current = Image.open(path3 + "example_moved.png")

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

    reconstructed = Image.fromarray(data.reconstruct(som.get_reconstructed_data(winners)))
    som_difference = ImageOps.autocontrast(ImageChops.difference(reconstructed, current).convert('L'))
    som_difference_modulated = ImageOps.autocontrast(ImageChops.multiply(som_difference, diff_winners))

    # dnf.input = np.divide(np.asarray(som_difference_modulated), 255)
    # dnf.update_map()
    # print(np.max(dnf.input))
    # print(np.min(dnf.input))
    #
    # print(np.max(dnf.potentials))
    # print(np.min(dnf.potentials))
    # dnf_display = Image.fromarray(np.multiply(dnf.potentials, 255)).convert('RGB')
    # som_difference.save("/users/yabernar/workspace/watSOM/Results/DNF/saliency"+str(i)+".png")
    # diff_winners.save("/users/yabernar/workspace/watSOM/Results/diff_winners"+str(i)+".png")

    # som_difference.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/tracking/som_difference.png" + "{0:02d}.png".format(i))
    # reconstructed.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/tracking/reconstructed.png" + "{0:02d}.png".format(i))
    # som_difference_modulated.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/tracking/som_difference_modulated.png" + "{0:02d}.png".format(i))
    # diff_winners.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/tracking/diff_winners.png" + "{0:02d}.png".format(i))
    # dnf_display.save("/users/yabernar/Documents/Presentation resources/Examples/last_run/tracking/dnf_display.png" + "{0:02d}.png".format(i))

    if plot is None:
        plot = []
        plt.subplot(3, 2, 1)
        plot.append(plt.imshow(current))
        plt.subplot(3, 2, 2)
        plot.append(plt.imshow(som_difference))
        plt.subplot(3, 2, 3)
        plot.append(plt.imshow(som_difference_modulated, cmap='gray'))
        plt.subplot(3, 2, 4)
        plot.append(plt.imshow(diff_winners))
        # plt.subplot(3, 2, 5)
        # plot.append(plt.imshow(dnf_display))

    else:
        plot[0].set_data(current)
        plot[1].set_data(som_difference)
        plot[2].set_data(som_difference_modulated)
        plot[3].set_data(diff_winners)
        # plot[4].set_data(dnf_display)
#        plot[3].set_data(truth)
    plt.pause(0.02)
    plt.draw()
plt.waitforbuttonpress()