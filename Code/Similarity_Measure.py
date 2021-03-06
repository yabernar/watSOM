import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from Code.Parameters import Parameters, Variable
from Code.SOM import SOM
from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14"
path2 = "/users/yabernar/workspace/aweSOM/Data/images/tracking/dogs/"

selected_category = 10
selected_element = 3

categories = sorted([d for d in os.listdir(path + '/dataset') if os.path.isdir(os.path.join(path, 'dataset', d))], key=str.lower)
elements = sorted([d for d in os.listdir(path + '/dataset/' + categories[selected_category]) if os.path.isdir(os.path.join(path, 'dataset', categories[selected_category],
                                                                                                           d))], key=str.lower)

print(categories)
print(elements)

chosen_path = os.path.join(path, "dataset", categories[selected_category], elements[selected_element])
temporal_ROI = (1000, 3200)
plot = None
# bkg = Image.open(chosen_path + "/input/" + 'in{0:06d}.jpg'.format(1))
bkg = Image.open(os.path.join(chosen_path, "input", "bkg.jpg"))
# bkg = Image.open(path2 + 'dogs{0:05d}.png'.format(1))


############
# TRACKING #
############
for i in range(temporal_ROI[0], temporal_ROI[1]):
    i *= 1
    print('Image ', i)
    current = Image.open(chosen_path + "/input/in{0:06d}.jpg".format(i))
    truth = Image.open(chosen_path + "/groundtruth/gt{0:06d}.png".format(i))
    # current = Image.open(path2 + "dogs{0:05d}.png".format(i))

    bgs_difference = ImageChops.difference(bkg, current).convert('L')
    thresh = 25
    fn = lambda x: 255 if x > thresh else 0
    bgs_thresholded = bgs_difference.convert('L').point(fn, mode='1')

    lbsp_diff = None
    image_euclidean = None

    if plot is None:
        plot = []
        plt.subplot(2, 2, 1)
        plot.append(plt.imshow(current))
        plt.subplot(2, 2, 2)
        plot.append(plt.imshow(truth))
        plt.subplot(2, 2, 3)
        plot.append(plt.imshow(bgs_difference))
        plt.subplot(2, 2, 4)
        plot.append(plt.imshow(bgs_thresholded))
    else:
        plot[0].set_data(current)
        plot[1].set_data(truth)
        plot[2].set_data(bgs_difference)
        plot[3].set_data(bgs_thresholded)
    plt.pause(0.02)
    plt.draw()
plt.waitforbuttonpress()