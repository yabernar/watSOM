import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

path = "/users/yabernar/GrosDisque/CDNET14"
path2 = "/users/yabernar/workspace/aweSOM/Data/images/tracking/dogs/"

categories = sorted(os.listdir(path + "/dataset"), key=str.lower)
elements = sorted(os.listdir(path + "/dataset/" + categories[3]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[3] + "/" + elements[0]
temporal_ROI = (1, 3200)
plot = None
bkg = Image.open(chosen_path + "/input/" + 'in{0:06d}.jpg'.format(1))
# bkg = Image.open(path2 + 'dogs{0:05d}.png'.format(1))


for i in range(temporal_ROI[0], temporal_ROI[1]):
    #    print('Epoch ', i)
    current = Image.open(chosen_path + "/input/in{0:06d}.jpg".format(i))
    truth = Image.open(chosen_path + "/groundtruth/gt{0:06d}.png".format(i))
    #    current = Image.open(path2 + "dogs{0:05d}.png".format(i))
    difference = ImageChops.difference(bkg, current).convert('L')
    # difference = difference.convert('L')
    # difference = np.array(difference)
    # difference = np.sum(difference, axis=2)
    # difference = np.divide(difference, 255*3)
    # difference = np.divide(difference, 255)
    # print(np.mean(np.square(difference)))
    if plot is None:
        plot = []
        plt.subplot(2, 2, 1)
        plot.append(plt.imshow(bkg))
        plt.subplot(2, 2, 2)
        plot.append(plt.imshow(current))
        plt.subplot(2, 2, 3)
        plot.append(plt.imshow(difference, cmap='gray'))
        plt.subplot(2, 2, 4)
        plot.append(plt.imshow(truth))
    else:
        plot[1].set_data(current)
        plot[2].set_data(difference)
        plot[3].set_data(truth)
    plt.pause(0.01)
    plt.draw()
plt.waitforbuttonpress()