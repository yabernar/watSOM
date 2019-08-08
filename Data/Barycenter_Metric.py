import os

from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14/dataset/nightVideos/bridgeEntry/groundtruth/"
files = sorted(os.listdir(path), key=str.lower)

plot = None
results = pd.DataFrame(columns=list('xy'))

###########
# DISPLAY #
###########
print(len(files))
for im in range(2450, len(files)):
    image = Image.open(path + files[im])

    array = np.divide(np.array(image), 255)
    array[array < 0.8] = 0
    barycentre = ndimage.measurements.center_of_mass(array)
    if np.isnan(barycentre[0]):
        tmp = np.divide(array.shape, 2)
        barycentre = (tmp[0], tmp[1])
    vecteur_distances = np.zeros(int(np.sqrt(array.shape[0]**2 + array.shape[1]**2)))
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            vecteur_distances[int(np.sqrt((x-int(barycentre[0]))**2 + (y-int(barycentre[1]))**2))] += array[x, y]
    vecteur_distances = np.divide(vecteur_distances, np.sum(vecteur_distances))
    print('Image ', im, 'Barycentre : ', barycentre)
    print(np.array(barycentre))
    print(pd.DataFrame(barycentre, columns=list('xy')))
    results.append(pd.DataFrame(barycentre))

    bary = (int(barycentre[0]), int(barycentre[1]))
    if plot is None:
        plot = []
        plt.subplot(1, 2, 1)
        plot.append(plt.imshow(array, cmap='gray', vmin=0, vmax=1, extent=[0, array.shape[1], array.shape[0], 0]))
        plot.append(plt.plot(bary[1], bary[0], 'b+'))
        plt.subplot(1, 2, 2)
        plot.append(plt.plot(vecteur_distances))
    else:
        plot[0].set_data(array)
        plot[1][0].set_xdata(bary[1])
        plot[1][0].set_ydata(bary[0])
        plot[2][0].set_ydata(vecteur_distances)
    plt.pause(0.01)
    plt.draw()
plt.waitforbuttonpress()

print(results.head())
