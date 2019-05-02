import os
from PIL import Image
import matplotlib.pyplot as plt

from Code.Parameters import Parameters
from Data.Mosaic_Image import MosaicImage

path = "/users/yabernar/GrosDisque/CDNET14"

categories = sorted(os.listdir(path+"/dataset"), key=str.lower)
elements = sorted(os.listdir(path+"/dataset/"+categories[1]), key=str.lower)
print(categories)
print(elements)

chosen_path = path + "/dataset/" + categories[1] + "/" + elements[1] + "/input/"
temporal_ROI = (570, 2050)
images_list = []
parameters = Parameters({"pictures_dim": [10, 10]})
plot = None
for i in range(temporal_ROI[0], temporal_ROI[1]+1):
    print('Opening image {0:06d}'.format(i))
    img = Image.open(chosen_path+'in{0:06d}.jpg'.format(i))
    data = MosaicImage(img, parameters)
    if plot is None:
        plot = plt.imshow(img)
    else:
        plot.set_data(img)
    plt.pause(0.1)
    plt.draw()
