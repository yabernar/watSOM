import os
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
output_path = "/users/yabernar/GrosDisque/CDNET14/saliency_as"
pawcs_path = "/users/yabernar/GrosDisque/CDNET14/PAWCS"


selected_category = 1
selected_element = 0

categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
categories_path = os.path.join(cdnet_path, categories[selected_category])

elements = sorted([d for d in os.listdir(categories_path) if os.path.isdir(os.path.join(categories_path, d))], key=str.lower)
elements_path = os.path.join(categories_path, elements[selected_element])
results_path = os.path.join(output_path, 'results', categories[selected_category], elements[selected_element])
supplements_path = os.path.join(output_path, 'supplements', categories[selected_category], elements[selected_element])
pawcs_path = os.path.join(pawcs_path, "results", categories[selected_category], elements[selected_element])

roi_file = open(os.path.join(elements_path, "temporalROI.txt"), "r").readline().split()
temporal_roi = (int(roi_file[0]), int(roi_file[1]))

plot = None
bkg = Image.open(os.path.join(elements_path, "input", "bkg.jpg"))


############
# TRACKING #
############
for i in range(temporal_roi[0], temporal_roi[1], 10):
    base = Image.open(elements_path + "/input/in{0:06d}.jpg".format(i))
    truth = Image.open(elements_path + "/groundtruth/gt{0:06d}.png".format(i))
    pawcs = Image.open(os.path.join(pawcs_path, "bin{0:06d}.png".format(i)))

    result = Image.open(os.path.join(results_path, 'bin{0:06d}.png'.format(i)))
    saliency = Image.open(os.path.join(supplements_path, 'saliency', 'sal{0:06d}.png'.format(i)))
    diff_winners = Image.open(os.path.join(supplements_path, 'diff_winners', 'win{0:06d}.png'.format(i)))
    difference = Image.open(os.path.join(supplements_path, 'difference', 'dif{0:06d}.png'.format(i)))

    bgs_difference = ImageChops.difference(bkg, base).convert('L')
    thresh = 5
    fn = lambda x: 255 if x > thresh else 0
    bgs_thresholded = bgs_difference.convert('L').point(fn, mode='1')


    if plot is None:
        plot = []
        plt.subplot(3, 3, 1)
        plot.append(plt.imshow(pawcs))
        plt.subplot(3, 3, 2)
        plot.append(plt.imshow(truth))
        plt.subplot(3, 3, 3)
        plot.append(plt.imshow(result))
        plt.subplot(3, 3, 4)
        plot.append(plt.imshow(base))
        plt.subplot(3, 3, 5)
        plot.append(plt.imshow(bgs_difference))
        plt.subplot(3, 3, 6)
        plot.append(plt.imshow(bgs_thresholded))
        plt.subplot(3, 3, 7)
        plot.append(plt.imshow(difference))
        plt.subplot(3, 3, 8)
        plot.append(plt.imshow(diff_winners))
        plt.subplot(3, 3, 9)
        plot.append(plt.imshow(saliency))
    else:
        plot[0].set_data(pawcs)
        plot[1].set_data(truth)
        plot[2].set_data(result)
        plot[3].set_data(base)
        plot[4].set_data(bgs_difference)
        plot[5].set_data(bgs_thresholded)
        plot[6].set_data(difference)
        plot[7].set_data(diff_winners)
        plot[8].set_data(saliency)
    plt.pause(0.01)
    plt.draw()
plt.waitforbuttonpress()